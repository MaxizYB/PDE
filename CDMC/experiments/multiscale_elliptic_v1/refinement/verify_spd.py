"""Scientific check of the matrix-defined core beyond diffusion M-matrices."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core import spectral_upper,fixed_filter,calibrate_dictionary,calibrated_update
from polynomial_joint import FineSparseAction
from coupled_coarse import apply_modes


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    rows=[]
    for i in range(20):
        n=25
        q=torch.linalg.qr(torch.randn(n,n,dtype=torch.float64))[0]
        eigen=torch.logspace(-4,1,n,dtype=torch.float64)
        a=q@torch.diag(eigen)@q.T
        matrix=a.to_sparse_csr()
        action=FineSparseAction(SimpleNamespace(prolongations=[],restrictions=[],level_sizes=[n]))
        action.matrix=matrix
        context=SimpleNamespace(compiled=SimpleNamespace(action=action))
        target=torch.randn(n,dtype=torch.float64)
        initial=torch.randn(n,dtype=torch.float64)
        system=SimpleNamespace(rhs=a@target,levels=[SimpleNamespace(diagonal_inverse=1/a.diag())])
        upper=spectral_upper(system,action)
        invroot=a.diag().rsqrt()
        largest=float(torch.linalg.eigvalsh(invroot[:,None]*a*invroot[None,:])[-1])
        filtered=fixed_filter(system,action,None,initial,16)
        modes=torch.randn(n,7,dtype=torch.float64)
        prepared=calibrate_dictionary(modes,context,system,None,SimpleNamespace(harmonic_steps=2))
        corrected=apply_modes(prepared,system,None,filtered,context)
        final=fixed_filter(system,action,None,corrected,8,.05)
        def energy(x):
            e=target-x
            return float(e@(a@e))
        values=[energy(x) for x in (initial,filtered,corrected,final)]
        debiased=[calibrated_update(prepared,system,None,filtered,context,t) for t in (1,2,4)]
        assert torch.equal(debiased[0],corrected)
        debiased_energy=[energy(x) for x in debiased]
        assert all(y<=x*(1+1e-10) for x,y in zip(debiased_energy,debiased_energy[1:]))
        assert upper>=largest
        assert all(y<=x*(1+1e-10) for x,y in zip(values,values[1:]))
        rows.append(dict(index=i,non_M_matrix=bool((a-torch.diag(a.diag())>1e-12).any()),
            certified_upper=upper,true_largest_normalized_eigenvalue=largest,energy_squared=values,
            small_system_sweeps=[1,2,4],debiased_energy_squared=debiased_energy))
    out=Path(sys.argv[1])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(samples=rows,passed=20,
        scope='FP64 algebraic validation on non-M-matrix SPD systems of spectral condition 1e5; checks spectral bound and energy decrease, not PDE generalization or field/flux monotonicity.'),indent=2)+'\n')
    print('20/20 non-M-matrix SPD core checks passed')


if __name__=='__main__':
    main()
