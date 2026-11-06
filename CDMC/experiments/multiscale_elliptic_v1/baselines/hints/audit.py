"""Numerically compare the adapter to the pinned official forward pass."""
import importlib.util
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from model import HintsDeepONet


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    vendor = Path(__file__).parent/'vendor/HINTS_numpy'
    sys.path.insert(0,str(vendor))
    import configs
    configs.DIMENSIONS=2
    configs.PROBLEM='poisson'
    configs.L_SHAPED=False
    configs.DEVICE=torch.device('cuda:0')
    configs.K_0=3.
    configs.K_SIGMA=2.
    spec=importlib.util.spec_from_file_location('official_deeponet',vendor/'deeponet.py')
    official=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    ref=official.DeepONet(np.linspace(0,1,31),logging.getLogger(),y_nodes=np.linspace(0,1,31))
    ours=HintsDeepONet(3.,2.,1.).to(configs.DEVICE)
    ours.branch_net.load_state_dict(ref.branch_net.state_dict())
    ours.trunk_net.load_state_dict(ref.trunk_net.state_dict())
    k=2.+torch.rand(3,31,31,device=configs.DEVICE)
    f=torch.randn_like(k)
    axis=torch.linspace(0,1,31,device=configs.DEVICE)
    coordinates=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    with torch.no_grad():
        expected=ref(k,f)[0].reshape(3,-1)
        actual=ours(k,f,coordinates)
        relative=float((actual-expected).norm()/expected.norm())
        scaling=float((ours(k,2*f,coordinates)-2*actual).abs().max())
    assert relative < 1e-5, relative
    assert scaling < 1e-6, scaling
    report=dict(official_forward_relative_difference=relative,
                positive_rhs_scaling_max_difference=scaling,
                parameters=sum(p.numel() for p in ours.parameters()),
                official_parameters=sum(p.numel() for p in ref.parameters()),
                note='Official unused scalar bias is absent in adapter; all active branch/trunk parameters match.')
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+chr(10))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
