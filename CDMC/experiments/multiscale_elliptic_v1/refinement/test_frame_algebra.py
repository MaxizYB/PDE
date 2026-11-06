"""CPU NumPy checks for the space construction; no torch installation needed."""
import json
import numpy as np


def run():
    rng=np.random.default_rng(20261104)
    n,r,s=120,20,8
    q=np.linalg.qr(rng.normal(size=(n,r)))[0]
    u=rng.normal(size=(n,s))
    u-=q@(q.T@u)
    u/=np.linalg.norm(u)
    raw=rng.normal(size=(s,r))
    c=2*raw/(1+np.linalg.norm(raw))
    z=q+u@c
    eigenvalues=np.linalg.eigvalsh(z.T@z)
    identity_error=np.linalg.norm(z.T@z-(np.eye(r)+c.T@u.T@u@c))
    assert identity_error<1e-10
    assert eigenvalues[0]>1-1e-10 and eigenvalues[-1]<5+1e-10
    k=rng.normal(size=(n,n))
    a=k.T@k+np.eye(n)
    error=rng.normal(size=n)
    def project(basis):
        return basis@np.linalg.solve(basis.T@a@basis,basis.T@a@error)
    correction=project(z)
    energy_before=error@a@error
    residual=error-correction
    energy_after=residual@a@residual
    transform=np.linalg.qr(rng.normal(size=(r,r)))[0]@np.diag(np.geomspace(1,1e-3,r))
    invariance=np.linalg.norm(correction-project(z@transform))/np.linalg.norm(correction)
    assert energy_after<=energy_before*(1+1e-12)
    assert invariance<1e-6
    return dict(gram_identity_error=identity_error,eigenvalue_min=float(eigenvalues[0]),
        eigenvalue_max=float(eigenvalues[-1]),energy_reduction=float(1-energy_after/energy_before),
        exact_projection_coordinate_relative_difference=float(invariance),passed=True)


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
