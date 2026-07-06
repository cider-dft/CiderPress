# CIDER Local Hybrids: Conventions, Validation, and Implementation Notes

Status: post refactor/validation campaign (2026-07, branch `LocHyb_SOSCF`).
The definitive equations and factor derivation live in the module docstring
of `ciderpress/pyscf/hybrid_scf.py`; this document records the conventions,
the validation suite, the campaign findings, and open items.

## Conventions (training/SCF contract)

- Raw hybrid feature: `feat = -0.5*ek = +|eps_x^exact|  >= 0`, where
  `ek = -1/2 F^T A F`, `F = dm_eff . chi`, `dm_eff = (2/nspin) P_s`
  (P_total for RKS, 2*P_s per spin for UKS). Shared by training-data
  generation and SCF via `descriptors._hyb_desc_getter`.
- LDA-normalized feature (`DensityNormalizer(1/LDA_FACTOR, -4/3)`):
  NEGATIVE, UEG value -1, magnitudes up to ~1e3 in low-density regions.
  The trained models' transforms (`VMap(gamma=-0.001)`, "pole far right")
  and spline grids are built for this range.
- Energy: `E_hyb = xmix * sum_s (1/nspin) sum_g w_g alpha_s(g) * (-feat)`;
  `exx_energy_baseline` recovers `e_x = -feat` smoothly (no abs kink).
- K weight: `W_s = dE/d(feat_s)` from the evaluator chain (alpha_eff is
  implicit). Block K: `K += +1/2 sum_g w_g W(g) chi(g) (A_g F_g)`;
  finalize adds `0.25*nspin*(K+K^T)` pre-`hermi_sum` (which doubles it).
- A-tensor modes: `CiderNumIntMixin.hyb_atensor_mode` in
  {"auto", "incore", "blockwise"}; incore caches A across SCF iterations
  (`ni._hyb_cache`) and computes warm-iteration features by pure BLAS.

## Validation suite

- `ciderpress/pyscf/tests/test_local_hybrid.py` (fast, stub-based):
  A-tensor oracle identities (RKS/UKS, ~1e-10), PBE0 anchors, FD
  energy-potential consistency matrix (the factor arbiter), RKS==UKS
  (H2/H2O), incore==blockwise, H-atom==UHF (one-electron SIC-free).
- `ciderpress/pyscf/tests/test_local_hybrid_slow.py`
  (`CIDER_SLOW_TESTS=1`): trained-model FD (vb2/sdmx1/tiny), spin/mode
  identities, H2+ dissociation regression vs
  `tests/data/h2p_reference.json`.
- Runner: `sbatch soscf_test_logs/run_hybrid_tests.sbatch`.
- Benchmarks: `scripts/benchmark_local_hybrid.py` (JSONs in
  `soscf_test_logs/bench_phase{2,3}_{rks,uks}.json`).

## Campaign findings (documented numeric changes)

1. **exx-feature train/eval sign mismatch (fixed, commit 2142488)**:
   all local-hybrid SCF runs with the GMTK-era models evaluated alpha(r)
   100% OUT of the trained spline domain (SCF used +0.5*ek). The
   interpolation package returns inconsistent value/gradient pairs
   out-of-domain (systematic FD failures); the historical `-abs(...)`
   masked the energy sign while alpha stayed mis-evaluated. Post-fix
   H2+ dissociation deltas (sdmx1_diff model) grow with bond stretching:
   -4.0e-5 Ha (1.45 A), -1.1e-4 (2.02 A), -3.7e-4 (3.16 A) — the curve
   tail was distorted. Reference regenerated
   (`tests/data/h2p_reference.json`).
2. **Factor conventions confirmed correct**: the suspicious RKS-0.5 vs
   UKS-1.0 finalize asymmetry and dm scalings compensate exactly
   (hermi_sum doubling + evaluator 1/nspin); enforced by the oracle/FD
   suite at ~1e-10.
3. **Performance** (glycine/def2-svp/level-2, warm iterations):
   RKS incore 7.5-9.4 -> 1.86 s/call, UKS incore ~18 -> 3.8 s/call
   (A-tensor iteration cache + BLAS-only feature/K builds); UKS incore
   also no longer duplicates the A tensor per spin.

## Known issues / follow-ups

- **NLDF UKS spin-polarized FD inconsistency (~3e-3 rel)**: pre-existing,
  NOT hybrid-specific (non-hybrid vb2 control fails identically;
  evaluator-level chain is clean; NLDF-free hybrid passes UKS). Likely
  related to the long-standing `test_nldf test_same_spin_issue` failure.
  Tests marked expectedFailure. Separate work item.
- Blockwise mode still runs two SGX sweeps per iteration (eps via
  direct-JK, K via batch_nuc); a lean fused sweep is a documented
  follow-up, as are Laqua-style P/S-junction screening and
  orbital-occupation derivatives (`HybridPlan.get_occd`).
- Local-hybrid nuclear gradients neglect the nonlocal A-tensor response
  (warn-once added in rks_grad/uks_grad); exact gradients need the
  three-term Laqua Eq. 11.
- Newton/SOSCF (`mf.newton()`): the proxy fxc kernel omits the hybrid
  exchange response (warned); `mf.fd_response = True` includes it
  exactly but recomputes eps_exx per inner iteration.
