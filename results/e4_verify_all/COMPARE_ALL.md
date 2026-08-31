# E4 held-bound verification, FULL scope (review P4)

Same-seed rerun of **every legacy gated E4 run that is still a reported result**, under the corrected vector-valued Proposition-10 held bound

```
iota_t = Delta eta_t * Hbar_t + eta_max,t0 * (M_H * P_t + nu_H),   Delta eta_t = ||eta_t - eta_t0||_inf
```
i.e. `dh.probe(rho, kappa, eta_vec=eta)` / `dh.bounds(eta)` (`e4_gpt2_tta.py` default path), against the shipped runs, which used the scalar interface `dh.probe(rho, kappa)` / `dh.bounds(float(eta.max()))` -- forcing `Delta eta_t == 0` and reading `eta_max` at the CURRENT step instead of at the last probe.  The old path survives behind `--legacy-hold`.

Legacy provenance is the ABSENCE of the `legacy_hold` / `held_bound` / `gate_open_steps` fields, which entered with the fix.

Driver flags are identical for every pair (read back from the shipped JSONs): `--tokens-per-domain 512000 --max-steps 3000 --batch 2 --seq-len 256 --probe-every 100 --kw-eps 0.15 --lr 0.001 --meta-lr 0.4` with `--domain-order` as tabulated (K=20, gamma=0.9, rank=4 for the `cohg` arm / 0 for `cohg_r0`, M_H=50).

**Scope.** The gated arms are the only ones in scope: the held bound `(rho_t, kappa_t)` is consumed exclusively by `CoordGatedController.maybe_update` (through `est.step(...)` / `beta_col`), so it can only ever change a run whose lambda update is gated.  `fixed` never builds a certificate at all (`est is None`); `hd` uses `HDBaseline.update`, which never sees rho or kappa; `cohg_nogate` short-circuits to `rho, kappa = 1.0, 0.0` before the drift hold is queried and takes an ungated sign step.  Those three arms are therefore bit-identical under either code path by construction and are excluded from the rerun.

## Coverage

| | count |
|---|---|
| legacy gated runs in scope | 14 |
| verified (rerun present) | 14 |
| still on the GPU queue | 0 |

## Per-pair verdict

| # | arm | seed | domain order | legacy dir | rerun dir | gate_open_frac == | coord_open_frac == | openings old/new | opening step(s) old | opening step(s) new | max abs d(lambda) | loss bit-identical | max abs d(loss) | d(online_ppl) | hvp_total old/new | events old/new | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cohg_r0 | 0 | wiki,news,code | e4_v2 | e4_fix | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 2 | cohg_r0 | 1 | wiki,news,code | e4_v2 | e4_fix | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 3 | cohg_r0 | 2 | wiki,news,code | e4_v2 | e4_fix | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 4 | cohg_r0 | 3 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 5 | cohg_r0 | 4 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 6 | cohg_r0 | 5 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 7 | cohg_r0 | 6 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 8 | cohg_r0 | 7 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 9 | cohg(r=4) | 0 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4692/4692 | 0/0 | bit-identical |
| 10 | cohg(r=4) | 1 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4692/4692 | 0/0 | bit-identical |
| 11 | cohg(r=4) | 2 | wiki,news,code | e4_v2 | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4692/4692 | 0/0 | bit-identical |
| 12 | cohg_r0 | 0 | code,news,wiki | e4_orders | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 13 | cohg_r0 | 1 | code,news,wiki | e4_orders | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |
| 14 | cohg_r0 | 2 | code,news,wiki | e4_orders | e4_verify_all | yes | yes | 1/1 | in (0, 20] (20-step lam_hist grid) | steps [1] | 0 | yes | 0 | +0.00e+00 | 4100/4100 | 0/0 | bit-identical |

## Perplexity, log-loss and final lambda

| # | arm | seed | order | online_ppl legacy | online_ppl rerun | delta | mean_logloss legacy | mean_logloss rerun | steps | final lambda legacy | final lambda rerun |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cohg_r0 | 0 | wiki,news,code | 20.678234 | 20.678234 | +0.00e+00 | 3.02908165 | 3.02908165 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 2 | cohg_r0 | 1 | wiki,news,code | 20.680758 | 20.680758 | +0.00e+00 | 3.02920372 | 3.02920372 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 3 | cohg_r0 | 2 | wiki,news,code | 20.707146 | 20.707146 | +0.00e+00 | 3.03047886 | 3.03047886 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 4 | cohg_r0 | 3 | wiki,news,code | 20.619300 | 20.619300 | +0.00e+00 | 3.02622752 | 3.02622752 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 5 | cohg_r0 | 4 | wiki,news,code | 20.667299 | 20.667299 | +0.00e+00 | 3.02855271 | 3.02855271 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 6 | cohg_r0 | 5 | wiki,news,code | 20.622307 | 20.622307 | +0.00e+00 | 3.02637334 | 3.02637334 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 7 | cohg_r0 | 6 | wiki,news,code | 20.699187 | 20.699187 | +0.00e+00 | 3.03009442 | 3.03009442 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 8 | cohg_r0 | 7 | wiki,news,code | 20.728053 | 20.728053 | +0.00e+00 | 3.03148798 | 3.03148798 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 9 | cohg(r=4) | 0 | wiki,news,code | 20.678234 | 20.678234 | +0.00e+00 | 3.02908165 | 3.02908165 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 10 | cohg(r=4) | 1 | wiki,news,code | 20.680758 | 20.680758 | +0.00e+00 | 3.02920372 | 3.02920372 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 11 | cohg(r=4) | 2 | wiki,news,code | 20.707146 | 20.707146 | +0.00e+00 | 3.03047886 | 3.03047886 | 2999/2999 | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | [-6.5078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] |
| 12 | cohg_r0 | 0 | code,news,wiki | 20.906249 | 20.906249 | +0.00e+00 | 3.04004809 | 3.04004809 | 2999/2999 | [-7.3078, -6.5078, -7.3078, -6.5078, -6.5078, -6.5078] | [-7.3078, -6.5078, -7.3078, -6.5078, -6.5078, -6.5078] |
| 13 | cohg_r0 | 1 | code,news,wiki | 21.638752 | 21.638752 | +0.00e+00 | 3.07448580 | 3.07448580 | 2999/2999 | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] |
| 14 | cohg_r0 | 2 | code,news,wiki | 21.663691 | 21.663691 | +0.00e+00 | 3.07563762 | 3.07563762 | 2999/2999 | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] |

## Provenance

| # | arm | seed | order | legacy `held_bound` | legacy `legacy_hold` | rerun `held_bound` | rerun `legacy_hold` | rerun `domain_order` |
|---|---|---|---|---|---|---|---|---|
| 1 | cohg_r0 | 0 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 2 | cohg_r0 | 1 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 3 | cohg_r0 | 2 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 4 | cohg_r0 | 3 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 5 | cohg_r0 | 4 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 6 | cohg_r0 | 5 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 7 | cohg_r0 | 6 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 8 | cohg_r0 | 7 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 9 | cohg(r=4) | 0 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 10 | cohg(r=4) | 1 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 11 | cohg(r=4) | 2 | wiki,news,code | (absent -> scalar legacy) | (absent) | vector_prop10 | False | wiki,news,code |
| 12 | cohg_r0 | 0 | code,news,wiki | (absent -> scalar legacy) | (absent) | vector_prop10 | False | code,news,wiki |
| 13 | cohg_r0 | 1 | code,news,wiki | (absent -> scalar legacy) | (absent) | vector_prop10 | False | code,news,wiki |
| 14 | cohg_r0 | 2 | code,news,wiki | (absent -> scalar legacy) | (absent) | vector_prop10 | False | code,news,wiki |

## Summary

| verdict | pairs |
|---|---|
| bit-identical | 14 |
| gate-identical (fp round-off) | 0 |
| DIFFERS | 0 |
| pending | 0 |

**Every legacy gated E4 run reproduces bit-for-bit under the corrected vector-valued Proposition-10 bound.**  All 14 pairs agree on `gate_open_frac`, `coord_open_frac`, the number and location of the gate openings, the whole logged lambda trajectory, the per-step loss trace, `hvp_total` and `events`.  The scalar drift-hold shortcut was therefore *inert* on the entire reported E4 grid: the gate opens once, at t <= 20, and lambda is frozen thereafter, so `Delta eta_t = 0` between probes and `eta_max` at the current step equals `eta_max` at the last probe.  The published E4 numbers stand as reported.

This closes review item P4 at full scope: the 3-seed check in `results/e4_fix/COMPARE.md` is no longer the basis of the claim -- every legacy gated run that remains a reported result has been re-verified on its own seed.

<sub>generated by `code/experiments/compare_e4_verify_all.py`</sub>
