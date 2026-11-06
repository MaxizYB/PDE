"""Independent NumPy algebra checks; no torch or neural performance claims."""
import json
import numpy as np


def run():
    rng = np.random.default_rng(20261106)
    n, rank, extra = 120, 20, 8
    t = rng.normal(size=(n, n))
    a = t.T @ t + np.eye(n)
    q = np.linalg.qr(rng.normal(size=(n, rank)))[0]
    h = rng.normal(size=(n, extra))
    e = rng.normal(size=n)
    g = q.T @ a @ q
    w = h - q @ np.linalg.solve(g, q.T @ a @ h)
    def proj(z, error):
        return z @ np.linalg.solve(z.T @ a @ z, z.T @ a @ error)
    eq = e - proj(q, e)
    ec = eq - proj(w, eq)
    joint = e - proj(np.column_stack((q, h)), e)
    cross = np.linalg.norm(q.T @ a @ w) / np.linalg.norm(a @ w)
    identity = np.linalg.norm(ec - joint) / np.linalg.norm(joint)
    pythagoras = abs(eq @ a @ eq - ec @ a @ ec - (eq - ec) @ a @ (eq - ec)) / (eq @ a @ eq)
    assert cross < 1e-12 and identity < 1e-12 and pythagoras < 1e-12
    assert ec @ a @ ec <= eq @ a @ eq
    exact_reference_error = q[:, 0]
    rotated = q + w @ rng.normal(size=(extra, rank))
    rotation_loss = exact_reference_error - proj(rotated, exact_reference_error)
    assert rotation_loss @ a @ rotation_loss > 1e-8
    return dict(passed=True, normalized_A_cross=cross, joint_relative_difference=identity,
                energy_identity_relative_error=pythagoras,
                reference_energy=float(eq @ a @ eq), complement_energy=float(ec @ a @ ec),
                equal_rank_rotation_can_lose_reference=True)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
