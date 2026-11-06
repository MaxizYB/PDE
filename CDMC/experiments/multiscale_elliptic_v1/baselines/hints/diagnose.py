"""Check training/evaluation units and material/source sensor consistency."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import HintsDeepONet, sensors


def main(args):
    torch.set_num_threads(2)
    d=torch.device('cuda:0')
    saved=torch.load(args.run/'model.pt',map_location=d,weights_only=False)
    model=HintsDeepONet(**saved['model_options']).to(d).eval()
    model.load_state_dict(saved['state'])
    pairs=dict(np.load(args.run/'pairs/val.npz'))
    raw=dict(np.load(args.data/'val.npz'))
    xy=np.load(args.data/'mesh.npz')['points']
    n=int(round(np.sqrt(len(xy))))
    count=int(json.loads((args.run/'pairs/manifest.json').read_text())['rhs_per_material'])
    ids=torch.arange(len(pairs['rhs']),device=d)
    t={k:torch.as_tensor(v,device=d) for k,v in pairs.items()}
    with torch.no_grad():
        pred=model(t['coefficient'][t['material']],t['rhs'],t['coordinates'])
        cached=model(t['coefficient'][t['material'][:128]],t['rhs'][:128],
                     trunk=model.trunk(t['coordinates']))
        err=((pred-t['solution']).norm(dim=1)/t['solution'].norm(dim=1)).cpu().numpy()
        ks=sensors(torch.as_tensor(raw['coefficient'][:1].reshape(1,n,n),device=d))
        full=model(ks,torch.ones_like(ks),torch.as_tensor(xy,dtype=torch.float32,device=d))[0]
        target=torch.as_tensor(raw['solution'][0],device=d)
        sensor_diff=float((ks-t['coefficient'][:1]).abs().max())
        stride=(n-1)//64
        consistency=float((full.reshape(n,n)[::stride,::stride].flatten()-pred[0]).abs().max())
    report=dict(options=saved['model_options'],
        saved_best_validation=saved['best_validation'],saved_best_epoch=saved['best_epoch'],
        cached_batch_relative_error=float(((cached-t['solution'][:128]).norm(dim=1)/
                                           t['solution'][:128].norm(dim=1)).mean()),
        validation_mean=float(err.mean()),constant_source_mean=float(err[::count].mean()),
        random_source_mean=float(np.delete(err,np.arange(0,len(err),count)).mean()),
        coefficient_sensor_difference=sensor_diff,query_consistency_difference=consistency,
        full_field_error=float((full-target).norm()/target.norm()),
        prediction_range=[float(full.min()),float(full.max())],
        target_range=[float(target.min()),float(target.max())],
        train_label_range=[float(t['solution'][0].min()),float(t['solution'][0].max())])
    assert abs(report['validation_mean']-saved['best_validation']) < 1e-4, report
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+chr(10))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path)
    main(p.parse_args())
