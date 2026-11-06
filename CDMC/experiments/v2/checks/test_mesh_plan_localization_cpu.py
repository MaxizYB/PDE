"""CPU-only numerical regressions without importing/installing PyTorch.

Load the actual NumPy/SciPy function definitions from numerics.py with AST;
this keeps the no-GPU workstation independent of the ML environment.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def numerical_functions():
    source = Path(__file__).resolve().parents[1] / "solver" / "numerics.py"
    names = {"jacobi_spectral_bound", "aggregate_prolongation", "localize_prolongation",
             "drop_galerkin_entries", "build_hierarchy"}
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    module = ast.Module(body=[node for node in parsed.body
                             if isinstance(node, ast.FunctionDef) and node.name in names],
                        type_ignores=[])
    namespace = {"np": np, "sp": sp}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class LocalizedPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = numerical_functions()

    def test_signed_row_sum_and_own_aggregate(self):
        p = sp.csr_matrix([[0.8, 0.15, 0.1, -0.05], [-0.1, 0.8, 0.2, 0.1],
                           [0.1, -0.15, 0.8, 0.25]])
        own = np.array([0, 1, 2])
        q = self.functions["localize_prolongation"](p, own, 2)
        self.assertLessEqual(np.diff(q.indptr).max(), 2)
        np.testing.assert_allclose(q @ np.ones(4), p @ np.ones(4), atol=1e-14)
        self.assertTrue(all(q[i, own[i]] != 0 for i in range(3)))

    def test_inactive_cap_is_exact(self):
        p = sp.csr_matrix([[0.8, 0.2], [-0.1, 0.7]])
        q = self.functions["localize_prolongation"](p, np.array([0, 1]), 64)
        self.assertEqual((p - q).nnz, 0)

    def test_compensation_reserves_missing_own_column(self):
        p = sp.csr_matrix([[0.0, 0.3, 0.4, 0.1]])
        q = self.functions["localize_prolongation"](p, np.array([0]), 2)
        self.assertLessEqual(q.nnz, 2)
        np.testing.assert_allclose(q.sum(axis=1), p.sum(axis=1), atol=1e-14)

    def test_exact_galerkin_and_positive_definiteness(self):
        n = 81
        matrix = sp.diags([-np.ones(n-1), 2.1*np.ones(n), -np.ones(n-1)],
                          offsets=[-1, 0, 1], format="csr")
        hierarchy = self.functions["build_hierarchy"](
            matrix, coarse_size=8, interpolation_max_entries=2)
        for i, entry in enumerate(hierarchy[:-1]):
            a, p, r = entry
            next_entry = hierarchy[i+1]
            coarse = next_entry[0] if isinstance(next_entry, tuple) else next_entry
            np.testing.assert_allclose((r @ a @ p).toarray(), coarse.toarray(), atol=1e-13)
            self.assertLessEqual(np.diff(p.indptr).max(), 2)
            np.linalg.cholesky(coarse.toarray())

    def test_zero_cap_preserves_default_hierarchy(self):
        matrix = sp.diags([-np.ones(63), 2.1*np.ones(64), -np.ones(63)],
                          offsets=[-1, 0, 1], format="csr")
        legacy = self.functions["build_hierarchy"](matrix, coarse_size=8)
        explicit = self.functions["build_hierarchy"](
            matrix, coarse_size=8, interpolation_max_entries=0)
        self.assertEqual(len(legacy), len(explicit))
        for a, b in zip(legacy, explicit):
            aa = a if isinstance(a, tuple) else (a,)
            bb = b if isinstance(b, tuple) else (b,)
            for x, y in zip(aa, bb):
                self.assertEqual((x - y).nnz, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
