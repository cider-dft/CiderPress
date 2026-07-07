# NLDF UKS Open-Shell FD Inconsistency: Diagnosis Report

Date: 2026-07-07 · Branch `LocHyb_SOSCF` @ 732b115 · Diagnostic driver:
`soscf_test_logs/diagnose_nldf_uks.py` · Regression test:
`ciderpress/pyscf/tests/test_nldf_scf.py`

## Verdict: M2 — a one-argument bug in `nr_uks`, root cause proven

**`nr_uks` calls `ni.nldfgen.get_potential(vxc_nldf_full[s, idm])` for BOTH
spin channels without the `spin` argument** (`ciderpress/pyscf/numint.py`,
NLDF potential application, currently lines 708-709). `get_potential`
(`ciderpress/dft/lcao_nldf_generator.py:223`) defaults `spin=0` and uses
the per-spin cached forward data `self._cache[spin]` (feature/exponent
derivatives from the matching `get_features(..., spin=s)` call). So the
**beta-channel NLDF potential is built from the alpha channel's cached
derivatives**.

Every symptom is explained: exact cancellation at spin symmetry
(closed-shell RKS≡UKS held at 1e-10), systematic delta-independent FD
error for open shells scaling with spin polarization, NLDF-only,
hybrid-and-non-hybrid alike, generator-level tests unaffected (they pass
`spin=s` explicitly), evaluator-level clean.

## Evidence chain

1. **A1 linear-probe** (smooth `f=w·X1` evaluator on the vb2 settings — no
   spline domains): FD through `nr_rks` clean (1e-9..1.6e-7, proper
   delta^2 scaling); through `nr_uks` on H2O+ systematically wrong
   (2.3e-5..5.6e-4, constant in delta) → chain-derivative bug, not a
   distribution/domain effect.
2. **Code**: `get_potential(self, vfeat, spin=0, ...)` → `cache =
   self._cache[spin]`; `nr_uks` omits `spin`; **the gradient module does
   it right** (`uks_grad.py:276-277, 462-468` pass `spin=0/1` explicitly —
   which is why `test_uks_grad` passes and confirms the intended API).
   All other callers audited: only the two `nr_uks` lines are wrong.
3. **Runtime confirmation** (no code change): a shim cycling `spin=0,1`
   across the two `get_potential` calls per set collapses the open-shell
   linear-probe FD error to 9e-8..1.4e-6 with proper delta^2 scaling.
4. **A2 domain check** (real vb2 model): minor boundary-touching spline
   violations (0.25–1%) exist but EQUALLY for closed and open shells →
   not the FD culprit (secondary observation only; low-density tails at
   the VMap edge).
5. Train/eval NLDF descriptor parity is covered by the passing
   `test_nldf_equivalence` (incl. UKS). (The quick A3 script's apparent
   mismatch is a driver artifact of the script, superseded by that test.)

## Measured impact of the fix (runtime class-patch, converged UKS SCF)

| system | E(buggy) | E(fixed) | dE (Ha) | max|d(dm)| |
|---|---|---|---|---|
| H2O+ (doublet) | -75.7833353711 | -75.7833791296 | **-4.4e-05** | 4.4e-03 |
| O2 (triplet)   | -149.9581811359 | -149.9583038441 | **-1.2e-04** | 5.9e-03 |
| H2+ (1e, beta empty) | -0.5666688420 | -0.5666688420 | +9e-15 | 4e-13 |

Fixed energies are LOWER (variationally consistent: the correct potential
converges to the true minimum of the unchanged energy functional — the
functional itself was always right; only the Fock matrix was corrupted).

**Campaign implications if fixed**: every open-shell system in the
GSCDB/AETM validations shifts by ~1e-5..1e-4 Ha (0.03–0.08 kcal/mol
per species). Since atomization energies difference open-shell atoms
against (mostly closed-shell) molecules, AE reaction energies shift
by roughly the sum of per-atom effects — potentially ~0.1–0.3 kcal/mol
per reaction, material at the ~1 kcal/mol accuracy scale. Closed-shell
results are provably unaffected. H2+ dissociation references unaffected
(empty beta channel).

## Proposed fix (pending go/no-go)

One line each in `ciderpress/pyscf/numint.py` (NLDF potential
application in `nr_uks`):

```python
wv_full[0, idm] += ni.nldfgen.get_potential(vxc_nldf_full[0, idm], spin=0)
wv_full[1, idm] += ni.nldfgen.get_potential(vxc_nldf_full[1, idm], spin=1)
```

Then: un-xfail `test_nldf_scf.py::test_fd_uks_open_shell` and the three
`test_local_hybrid_slow.py` FD xfails (vb2-uks / h2plus / nonhyb-uks
control — h2plus may need its tolerance revisited given the empty-channel
FD caveat); full suite on the test partition; document the numeric shift.

## Secondary finding: `TestNLDFGaussian::test_same_spin_issue` is FLAKY

Reproduced pass/fail/pass across identical invocations on one node
(historically failed at ~5.4e-4 vs rtol 1e-4; 6/6 passes in later
repeats at both OMP_NUM_THREADS=1 and 16). The test converges two
symmetry-broken O2+ UHF solutions to 1e-13 and compares descriptors at
rtol=1e-4 — run-to-run variation likely reflects UHF solution
sensitivity (DIIS/threading nondeterminism) rather than a descriptor
physics bug; the Spline twin's margin is larger. Recommended fix:
stabilize the test (deterministic init guess or newton() convergence for
the two UHF references, and/or set the tolerance from a measured
run-to-run spread), not C-code archaeology — unless stabilized UHF
still shows spread, which would then indicate a real race.
