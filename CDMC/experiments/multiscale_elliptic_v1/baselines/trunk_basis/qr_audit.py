"""Validation-only audit: distinguish space quality from ridge conditioning.

Orthogonalization is an invertible change of coordinates in exact arithmetic.
A regularized Galerkin solve, however, is not coordinate invariant. This
checks the same learned space before/after QR without updating any weights.
It is an offline diagnostic; no inference speed is claimed for this variant.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate import (torch, DarcySplit, tensors, SparseCompiled, CoarseContext,
                      load_no, predict, load_dictionary, smooth, prepare, cycles,
                      summary, save_json)
from polynomial_joint import P1Norms


def main(args):
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    split=DarcySplit(args.data,'val')
    compiled=SparseCompiled(split.mesh,device)
    context=CoarseContext(compiled,1)
    norms=P1Norms(split.mesh,device)
    a,_,xy,_,mask=tensors(split,device)
    fno,stats,_=load_no(args.no_dir,device)
    model,saved=load_dictionary(args.dictionary,device)
    model.eval()
    opts=SimpleNamespace(**saved['options'])
    rows=[]
    with torch.no_grad():
        for i in range(args.count):
            full=predict(fno,a[i:i+1],xy,mask,stats).flatten()
            target=torch.as_tensor(split.target[i],device=device)
            system,ae,_=compiled.build(a[i].flatten(),torch.zeros_like(full))
            initial=smooth(system,compiled.action,ae,full[system.free],opts.pre,opts.lower,opts)
            modes=model(context.features(system,ae,initial),context)
            q,r=torch.linalg.qr(modes.double(),mode='reduced')
            values={}
            for label,basis in [('original',modes),('same_space_QR',q.float())]:
                prepared=prepare(basis,context,system,ae,opts)
                out=cycles(initial,prepared,system,ae,context,opts,2,filtered=True)
                metrics=(100*norms.ratios(out,target,system,ae)).cpu().tolist()
                eig=torch.linalg.eigvalsh(prepared[3])
                values[label]=dict(zip(('L2_percent','gradient_percent','flux_L2_percent'),metrics))
                values[label]['normalized_gram_smallest_eigenvalue']=float(eig[0])
                values[label]['normalized_gram_largest_eigenvalue']=float(eig[-1])
            rows.append(dict(index=i,methods=values,qr_diagonal_min=float(r.diagonal().abs().min())))
    result=dict(split='val',count=len(rows),cycles=2,dictionary=str(args.dictionary),
        definition='Same pretrained learned span, no new training; all columns retained by QR; tests whether coordinate-sensitive ridge limits deployed correction.',
        samples=rows,summary={name:{key:summary([r['methods'][name][key] for r in rows])
                                  for key in values[name]} for name in values})
    save_json(args.out,result)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','no-dir','dictionary','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=8)
    main(p.parse_args())
