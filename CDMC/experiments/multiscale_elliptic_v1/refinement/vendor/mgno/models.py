"""Selected official MgNO definitions; see PROVENANCE.json."""
import torch
from torch import nn
class MgIte(nn.Module):
    """One multigrid smoothing iteration: u <- u + S(f - A(u)).

    Args:
        A: Learnable operator approximating the PDE operator (u -> f space).
        S: Learnable smoother (f -> u space).
    """

    def __init__(self, A, S):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, out):
        """Apply one smoothing step.

        Args:
            out: Either a tuple ``(u, f)`` for subsequent iterations, or just
                 ``f`` for the very first iteration (handled by MgIte_init).

        Returns:
            tuple: Updated ``(u, f)``.
        """
        if isinstance(out, tuple):
            u, f = out
            u = u + self.S(f - self.A(u))
        else:
            f = out
            u = self.S(f)
        return (u, f)

class MgIte_init(nn.Module):
    """Initial smoothing step when no previous approximation is available.

    Computes ``u = S(f)`` and packages the result as ``(u, f)``.

    Args:
        S: Learnable smoother (f -> u space).
    """

    def __init__(self, S):
        super().__init__()
        self.S = S

    def forward(self, f):
        u = self.S(f)
        return (u, f)

class Restrict(nn.Module):
    """Multigrid restriction operator: projects ``(u, f)`` to a coarser grid.

    Optionally computes the residual ``f - A(u)`` before restricting, which
    corresponds to the *full approximation scheme* (FAS) when ``A`` is given.

    Args:
        Pi: Prolongation/restriction operator for *u* (fine -> coarse).
        R:  Restriction operator for *f* (fine -> coarse).
        A:  Optional operator for residual computation (u -> f space).
    """

    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.A = A

    def forward(self, out):
        u, f = out
        if self.A is not None:
            f = self.R(f - self.A(u))
        else:
            f = self.R(f)
        u = self.Pi(u)
        return (u, f)

class MgConv_DC(nn.Module):
    """V-cycle MgConv block for Darcy-flow and pipe-flow benchmarks.

    Similar to :class:`MgConv` but omits layer normalization after
    prolongation, which works better for non-periodic, Dirichlet-type problems.

    Args:
        num_iteration: List of ``[num_pre, num_post]`` smoothing counts per level.
        num_channel_u: Channels for the solution field.
        num_channel_f: Channels for the forcing/rhs field.
        padding_mode: Convolution padding (default: ``'zeros'``).
        bias: Bias in smoothing convolutions.
        use_res: Use residual-based restriction.
    """

    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='zeros', bias=False, use_res=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        # Prolongation (transposed conv) for each level transition
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u,
                               kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(len(num_iteration) - 1)
        ])

        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            post_smooth_layers = []

            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                if l == 0 and i == 0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))

            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer' + str(l),
                    nn.Sequential(*post_smooth_layers))

            if l < len(num_iteration) - 1:
                A = nn.Conv2d(num_channel_u, num_channel_f,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                Pi = nn.Conv2d(num_channel_u, num_channel_u,
                               kernel_size=3, stride=2, padding=1,
                               bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f,
                              kernel_size=3, stride=2, padding=1,
                              bias=False, padding_mode=padding_mode)
                layers = [Restrict(Pi, R, A) if use_res else Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        """V-cycle forward pass without LayerNorm on prolongation."""
        out_list = [0] * len(self.num_iteration)
        out = f

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out)
            out_list[l] = out

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j + 1][0])
            out_list[j] = getattr(self, 'post_smooth_layer' + str(j))((u_post, f))

        return out_list[0][0]

def _build_activation(activation: str) -> nn.Module:
    """Return a PyTorch activation module by name."""
    activation_map = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'silu': nn.SiLU(),
    }
    if activation not in activation_map:
        raise NameError(f"Invalid activation '{activation}'. "
                        f"Choose from {list(activation_map)}")
    return activation_map[activation]

class MgNO_DC(nn.Module):
    """MgNO model for Darcy-flow and pipe-flow benchmarks.

    Each layer consists of a parallel MgConv_DC branch (multi-scale nonlocal)
    and a pointwise 1×1 convolution (local), whose outputs are summed before
    the nonlinearity.

    Args:
        num_layer: Number of stacked MgConv_DC + 1×1 blocks.
        num_channel_u: Feature channels.
        num_channel_f: Input channels.
        num_classes: Unused; kept for API consistency.
        num_iteration: Per-level smoothing counts for MgConv_DC.
        in_chans: Number of input channels (default: 1).
        normalizer: Optional output normalizer with a ``decode`` method.
        output_dim: Number of output channels.
        activation: Nonlinearity name.
        padding_mode: Convolution padding mode.
    """

    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes,
                 num_iteration, in_chans=1, normalizer=None, output_dim=1,
                 activation='gelu', padding_mode='zeros'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([
            MgConv_DC(num_iteration, num_channel_u, num_channel_f,
                      padding_mode=padding_mode)
        ])
        self.linear_list = nn.ModuleList([
            nn.Conv2d(num_channel_f, num_channel_u,
                      kernel_size=1, stride=1, padding=0, bias=True)
        ])
        for _ in range(num_layer - 1):
            self.conv_list.append(
                MgConv_DC(num_iteration, num_channel_u, num_channel_u,
                          padding_mode=padding_mode)
            )
            self.linear_list.append(
                nn.Conv2d(num_channel_u, num_channel_u,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        self.normalizer = normalizer
        self.act = _build_activation(activation)

    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u) + self.linear_list[i](u))
        u = self.linear(u)
        return self.normalizer.decode(u) if self.normalizer else u
