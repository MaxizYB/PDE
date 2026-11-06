"""Frozen predictor interface: the correction algebra does not depend on NO type."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from vendor.mgno.models import MgNO_DC


class MgNOPredictor(nn.Module):
    def __init__(self, width=24, depth=4):
        super().__init__()
        self.model = MgNO_DC(num_layer=depth, num_channel_u=width, num_channel_f=1,
            num_classes=1, num_iteration=[[1,1],[1,1],[1,1],[2,0]], activation='gelu')

    def forward(self, values, coordinates):
        h,w = values.shape[-2:]
        # Official DC prolongation doubles sizes; pad to its 8-cell hierarchy.
        x = F.pad(values[:,None], (0,(-w)%8,0,(-h)%8))
        return self.model(x)[:,0,:h,:w]


class DCNOPredictor(nn.Module):
    """Pinned DCNO blocks; scalar structured-grid input/output adapter."""

    def __init__(self, resolution=129, width=32):
        super().__init__()
        from vendor.dcno.models import DCNO2d
        self.model = DCNO2d(dict(feature_dim=width, res_output=resolution,
            FNO_padding=5, patch_size=3, in_dim=1, subsample_stride=1,
            patch_padding=1, modes=12, num_spectral_layers=3,
            dilation=[1, 3, 9], mlp_hidden_dim=128))

    def forward(self, values, coordinates):
        return self.model(values[:, None]).reshape_as(values)


class M2NOPredictor(nn.Module):
    """Official Darcy architecture, padded for its five dyadic levels."""

    def __init__(self, resolution=129):
        super().__init__()
        from vendor.m2no.m2no2d import M2NO2d
        self.padded_resolution = ((resolution+31)//32)*32
        self.model = M2NO2d(in_channels=3, out_channels=1, k=4, c=2,
            num_layers=5, grid_levels=[3, 3, 2, 2, 1], base='legendre',
            bias=True, padding_mode='circular', norm=True,
            resolutions=[self.padded_resolution]*2, activation='gelu')

    def forward(self, values, coordinates):
        h, w = values.shape[-2:]
        x = torch.cat((values[..., None],
                       coordinates.expand(len(values), -1, -1, -1)), -1)
        x = F.pad(x.permute(0, 3, 1, 2),
                  (0, self.padded_resolution-w, 0, self.padded_resolution-h))
        return self.model(x.permute(0, 2, 3, 1))[:, :h, :w, 0]


PREDICTORS = {'MgNO': MgNOPredictor, 'DCNO': DCNOPredictor,
              'M2NO': M2NOPredictor}


def load_predictor(path, device):
    from run import load_no
    saved = torch.load(path/'no.pt', map_location=device, weights_only=False)
    kind = saved.get('predictor_type', 'FNO')
    if kind == 'FNO':
        return load_no(path, device)
    if kind not in PREDICTORS:
        raise ValueError(f'Unknown frozen predictor {kind!r}; register an explicit adapter.')
    model = PREDICTORS[kind](**saved['model_options']).to(device)
    model.load_state_dict(saved['state'])
    model.eval()
    return model, saved['stats'], saved
