#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

"""
Unit tests for OmegaMap with NST mode support.

This test suite verifies:
1. NST -> NPA conversion formulas are correct
2. Gradients are correct using finite difference
3. Backward compatibility with NPA mode
4. Serialization works correctly
"""

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from ciderpress.dft.transform_data import FeatureList, OmegaMap


class TestOmegaMapNST(unittest.TestCase):
    """Test suite for OmegaMap with NST mode."""

    def setUp(self):
        """Set up test data."""
        # Physical constants (must match transform_data.py)
        self.CFC = 0.3 * (3 * np.pi**2) ** (2.0 / 3)
        self.CFAC = 4 * (3 * np.pi**2) ** (2.0 / 3)
        
        # Test parameters for Omega
        self.c = 0.1
        self.B = 0.5
        self.C = 0.5
        
        # Test data: realistic density values
        # n: density (a.u.^-3)
        # sigma: |∇n|² (a.u.^-8) for NST, or s² (dimensionless) for NPA
        # tau: kinetic energy density (a.u.^-5) for NST, or α (dimensionless) for NPA
        self.test_vec_nst = np.array([
            [1.0, 0.5, 0.6],    # typical valence region
            [0.1, 0.01, 0.05],  # low density region
            [10.0, 50.0, 80.0], # core region
            [0.01, 1e-5, 1e-3], # very low density
        ]).T  # Shape: (3, n_samples)
        
        # Corresponding NPA features (manually converted for testing)
        self.test_vec_npa = self._convert_nst_to_npa(self.test_vec_nst)

    def _convert_nst_to_npa(self, vec_nst):
        """Convert NST features [n, σ, τ] to NPA features [n, s², α]."""
        n = vec_nst[0]
        sigma = vec_nst[1]
        tau = vec_nst[2]
        
        # s² = σ / (CFAC * n^(8/3))
        s2 = sigma / (self.CFAC * np.power(n, 8.0 / 3))
        
        # α = (τ - σ/(8n)) / (CFC * n^(5/3))
        alpha = (tau - sigma / (8 * n)) / (self.CFC * np.power(n, 5.0 / 3))
        
        return np.stack([n, s2, alpha])

    def test_nst_conversion_formula(self):
        """Test that NST -> NPA conversion formulas are correct."""
        omega_nst = OmegaMap(
            i_n=0, i_s=1, i_alpha=2, 
            c=self.c, B=self.B, C=self.C, 
            slmode="nst"
        )
        omega_npa = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=self.c, B=self.B, C=self.C,
            slmode="npa"
        )
        
        # Compute Omega for NST and NPA inputs
        y_nst = np.zeros(self.test_vec_nst.shape[1])
        y_npa = np.zeros(self.test_vec_npa.shape[1])
        
        omega_nst.fill_feat_(y_nst, self.test_vec_nst)
        omega_npa.fill_feat_(y_npa, self.test_vec_npa)
        
        # They should give the same result
        assert_allclose(
            y_nst, y_npa, rtol=1e-6, atol=1e-10,
            err_msg="NST and NPA modes should produce the same Omega values"
        )

    def test_npa_backward_compatibility(self):
        """Test that NPA mode behaves identically to the original implementation."""
        # Create OmegaMap with explicit slmode="npa"
        omega_explicit = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=self.c, B=self.B, C=self.C,
            slmode="npa"
        )
        
        # Create OmegaMap with default slmode (should be "npa")
        omega_default = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=self.c, B=self.B, C=self.C
        )
        
        y_explicit = np.zeros(self.test_vec_npa.shape[1])
        y_default = np.zeros(self.test_vec_npa.shape[1])
        
        omega_explicit.fill_feat_(y_explicit, self.test_vec_npa)
        omega_default.fill_feat_(y_default, self.test_vec_npa)
        
        assert_allclose(
            y_explicit, y_default, rtol=1e-15,
            err_msg="Default slmode should be 'npa'"
        )

    def test_gradient_npa_mode(self):
        """Test gradients in NPA mode using finite difference."""
        flist = FeatureList([
            OmegaMap(i_n=0, i_s=1, i_alpha=2, c=self.c, B=self.B, C=self.C, slmode="npa")
        ])
        
        gradfd = self._get_grad_fd(self.test_vec_npa, flist, delta=1e-6)
        grada = self._get_grad_a(self.test_vec_npa, flist)
        
        # Use allclose with reasonable tolerances for numerical derivatives
        assert_allclose(
            gradfd, grada, rtol=1e-4, atol=1e-6,
            err_msg="NPA mode gradients should match finite difference"
        )

    def test_gradient_nst_mode(self):
        """Test gradients in NST mode using finite difference."""
        flist = FeatureList([
            OmegaMap(i_n=0, i_s=1, i_alpha=2, c=self.c, B=self.B, C=self.C, slmode="nst")
        ])
        
        gradfd = self._get_grad_fd(self.test_vec_nst, flist, delta=1e-6)
        grada = self._get_grad_a(self.test_vec_nst, flist)
        
        # NST mode involves more transformations, so allow slightly larger tolerance
        assert_allclose(
            gradfd, grada, rtol=1e-3, atol=1e-6,
            err_msg="NST mode gradients should match finite difference"
        )

    def test_serialization_with_slmode(self):
        """Test that slmode is correctly saved and loaded."""
        omega_nst = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=self.c, B=self.B, C=self.C,
            slmode="nst"
        )
        
        # Serialize
        d = omega_nst.as_dict()
        
        # Check that slmode is in the dict
        self.assertIn("slmode", d)
        self.assertEqual(d["slmode"], "nst")
        
        # Deserialize
        omega_loaded = OmegaMap.from_dict(d)
        
        # Check that slmode is preserved
        self.assertEqual(omega_loaded.slmode, "nst")
        
        # Check that behavior is the same
        y_original = np.zeros(self.test_vec_nst.shape[1])
        y_loaded = np.zeros(self.test_vec_nst.shape[1])
        
        omega_nst.fill_feat_(y_original, self.test_vec_nst)
        omega_loaded.fill_feat_(y_loaded, self.test_vec_nst)
        
        assert_allclose(y_original, y_loaded, rtol=1e-15)

    def test_serialization_backward_compatibility(self):
        """Test that old dicts without slmode still work (default to npa)."""
        # Simulate an old dict without slmode
        d = {
            "code": "Omega",
            "i_n": 0,
            "i_s": 1,
            "i_alpha": 2,
            "c": self.c,
            "B": self.B,
            "C": self.C,
        }
        
        omega = OmegaMap.from_dict(d)
        
        # Should default to "npa"
        self.assertEqual(omega.slmode, "npa")

    def test_edge_cases(self):
        """Test edge cases: very small density, NaN handling, etc."""
        # Test with very small density
        edge_vec_nst = np.array([
            [1e-12, 1e-20, 1e-18],  # extremely small density
            [0.0, 0.0, 0.0],        # zero (should be handled)
            [np.nan, 1.0, 1.0],     # NaN in density
            [1.0, np.nan, 1.0],     # NaN in sigma
            [1.0, 1.0, np.nan],     # NaN in tau
        ]).T
        
        omega_nst = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=self.c, B=self.B, C=self.C,
            slmode="nst"
        )
        
        y = np.zeros(edge_vec_nst.shape[1])
        
        # Should not crash
        omega_nst.fill_feat_(y, edge_vec_nst)
        
        # Results should be finite
        self.assertTrue(np.all(np.isfinite(y)))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        extreme_vec_nst = np.array([
            [1e-8, 1e-12, 1e-10],   # very low density
            [1e2, 1e6, 1e5],        # very high values
            [1.0, 1e-20, 1e-18],    # very small gradients
        ]).T
        
        omega_nst = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=self.c, B=self.B, C=self.C,
            slmode="nst"
        )
        
        y = np.zeros(extreme_vec_nst.shape[1])
        omega_nst.fill_feat_(y, extreme_vec_nst)
        
        # Should not have NaN or Inf
        self.assertTrue(np.all(np.isfinite(y)))
        
        # Values should be in reasonable range [0, 1) due to Omega formula
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y < 1))

    def test_physical_consistency(self):
        """Test that Omega values are physically reasonable."""
        omega_nst = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=self.c, B=self.B, C=self.C,
            slmode="nst"
        )
        
        y = np.zeros(self.test_vec_nst.shape[1])
        omega_nst.fill_feat_(y, self.test_vec_nst)
        
        # Omega should be in [0, 1) due to the formula: c*Ω/(1+c*Ω)
        self.assertTrue(np.all(y >= 0), "Omega should be non-negative")
        self.assertTrue(np.all(y < 1), "Omega should be less than 1")

    # ========== Helper Methods (CORRECTED TO MATCH ORIGINAL TEST LOGIC) ==========
    
    def _get_grad_fd(self, vec, feat_list, delta=1e-6):
        """
        Compute gradient using finite difference (matches original test logic).
        
        This computes the sum of gradients across all samples, which is what
        the original test suite expects.
        
        Returns:
            deriv (n_features, n_samples): gradient for each feature and sample
        """
        deriv = np.zeros(vec.shape)
        for i in range(vec.shape[0]):
            dvec = vec.copy()
            dvec[i] += delta  # Perturb all samples' i-th feature
            # Sum over output features (axis=0), keep per-sample gradients
            deriv[i] += np.sum((feat_list(dvec.T).T - feat_list(vec.T).T) / delta, axis=0)
        return deriv

    def _get_grad_a(self, vec, feat_list):
        """
        Compute gradient analytically (matches original test logic).
        
        Returns:
            deriv (n_features, n_samples): gradient for each feature and sample
        """
        deriv = np.zeros(vec.shape)
        feat = np.zeros((feat_list.nfeat, vec.shape[1]))
        feat_list.fill_vals_(feat, vec)
        fderiv = np.ones(feat.shape)  # Shape: (n_output_features, n_samples)
        feat_list.fill_derivs_(deriv, fderiv, vec)
        return deriv


class TestOmegaMapPhysics(unittest.TestCase):
    """Additional tests for physical correctness."""
    
    def test_uniform_electron_gas_limit(self):
        """Test behavior for uniform electron gas (σ=0, α≈1)."""
        CFC = 0.3 * (3 * np.pi**2) ** (2.0 / 3)
        
        # Uniform electron gas: σ=0, τ=CFC*n^(5/3)
        rho = 1.0
        sigma = 0.0
        tau_ueg = CFC * rho ** (5.0 / 3)
        
        vec_nst = np.array([[rho], [sigma], [tau_ueg]])
        
        omega_nst = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=0.1, B=0.5, C=0.5,
            slmode="nst"
        )
        
        y = np.zeros(1)
        omega_nst.fill_feat_(y, vec_nst)
        
        # For UEG: s²=0, α≈1, so Omega should be well-defined
        self.assertTrue(np.isfinite(y[0]))
        self.assertGreater(y[0], 0)

    def test_gradient_dominance_limit(self):
        """Test behavior when gradient term dominates."""
        # Large sigma compared to tau
        vec_nst = np.array([
            [1.0],
            [10.0],  # Large sigma
            [0.1],   # Small tau
        ])
        
        omega_nst = OmegaMap(
            i_n=0, i_s=1, i_alpha=2,
            c=0.1, B=0.5, C=0.5,
            slmode="nst"
        )
        
        y = np.zeros(1)
        omega_nst.fill_feat_(y, vec_nst)
        
        # Should still give reasonable result
        self.assertTrue(np.isfinite(y[0]))
        self.assertGreaterEqual(y[0], 0)
        self.assertLess(y[0], 1)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
