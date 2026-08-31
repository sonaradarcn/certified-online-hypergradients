# GPT-2 (E4, results/e4_v2): gate-open timing

`lam_hist` is sampled every 20 steps, so a lambda change between two samples localises a gate-open event to the half-open interval `(t_prev, t_cur]`. Domain boundaries are at t=1000 and t=2000 (3 domains: D1 t<1000, D2 1000<=t<2000, D3 t>=2000).

The stored `gate_open_frac` gives the exact number of accepted meta-updates: `gate_open_frac x steps`.

| arm | seed | gate_open_frac | exact #opens | detected change intervals | coords moved | delta lambda | domain |
|---|---|---|---|---|---|---|---|
| cohg_r0 (rank-0, gated) | 0 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_r0 (rank-0, gated) | 1 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_r0 (rank-0, gated) | 2 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_r0 (rank-0, gated) | 3 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_r0 (rank-0, gated) | 4 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_r0 (rank-0, gated) | 5 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_r0 (rank-0, gated) | 6 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_r0 (rank-0, gated) | 7 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg (rank-4, gated) | 0 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg (rank-4, gated) | 1 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg (rank-4, gated) | 2 | 0.000333444 | 1 | (0,20] | emb,h0-2,h3-5,h6-8,h9-11,ln_f | +0.40,+0.40,+0.40,+0.40,+0.40,+0.40 | D1 |
| cohg_nogate (gate forced open) | 0 | 1 | 2999 | 149 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |
| cohg_nogate (gate forced open) | 1 | 1 | 2999 | 142 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |
| cohg_nogate (gate forced open) | 2 | 1 | 2999 | 147 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |
| cohg_nogate (gate forced open) | 3 | 1 | 2999 | 148 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |
| cohg_nogate (gate forced open) | 4 | 1 | 2999 | 147 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |
| cohg_nogate (gate forced open) | 5 | 1 | 2999 | 141 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |
| cohg_nogate (gate forced open) | 6 | 1 | 2999 | 146 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |
| cohg_nogate (gate forced open) | 7 | 1 | 2999 | 145 intervals (every sample) | all 6 | +-0.40 per step | D1,D2,D3 |

## Per-seed summary (gated arms)

- `cohg_r0` seed 0: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg_r0` seed 1: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg_r0` seed 2: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg_r0` seed 3: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg_r0` seed 4: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg_r0` seed 5: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg_r0` seed 6: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg_r0` seed 7: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg` seed 0: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg` seed 1: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
- `cohg` seed 2: exact opens = 1 / 2999 steps (0.033% of steps); detected at [(0, 20)]; D1=1, D2=0, D3=0. Total lambda displacement per coord: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

## Ungated arm (cohg_nogate)

- seed 0: gate forced open every step (2999/2999); online PPL 19.773; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-13.816, -11.816, -13.816, -13.816, -2.303, -8.616]; per-coord range (max-min) [8.0, 11.513, 11.513, 11.513, 11.513, 11.2]; lambda at boundaries t=1000 -> [-13.816, -13.816, -5.903, -13.503, -11.016, -8.616], t=2000 -> [-13.816, -4.216, -13.816, -3.416, -13.016, -7.816]
- seed 1: gate forced open every step (2999/2999); online PPL 38.026; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-13.816, -13.816, -3.016, -3.103, -8.303, -6.216]; per-coord range (max-min) [11.513, 11.513, 11.513, 11.513, 11.513, 11.513]; lambda at boundaries t=1000 -> [-13.816, -13.416, -13.816, -13.816, -13.416, -10.616], t=2000 -> [-13.816, -13.816, -5.416, -3.103, -5.416, -3.103]
- seed 2: gate forced open every step (2999/2999); online PPL 20.231; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-13.816, -13.816, -13.816, -13.816, -13.816, -2.703]; per-coord range (max-min) [7.6, 11.513, 11.513, 11.513, 11.513, 11.513]; lambda at boundaries t=1000 -> [-13.816, -13.016, -13.816, -13.816, -13.816, -13.416], t=2000 -> [-13.416, -13.416, -12.616, -3.103, -2.303, -3.503]
- seed 3: gate forced open every step (2999/2999); online PPL 26.721; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-3.416, -13.503, -13.816, -2.303, -13.816, -13.816]; per-coord range (max-min) [10.4, 11.513, 11.513, 11.513, 11.513, 10.713]; lambda at boundaries t=1000 -> [-13.816, -13.816, -2.303, -13.816, -13.816, -13.416], t=2000 -> [-13.816, -11.816, -13.816, -3.903, -3.103, -8.216]
- seed 4: gate forced open every step (2999/2999); online PPL 23.720; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-13.816, -13.816, -13.816, -13.816, -13.816, -13.816]; per-coord range (max-min) [11.513, 11.113, 11.113, 11.513, 11.513, 11.513]; lambda at boundaries t=1000 -> [-13.816, -13.816, -13.816, -10.216, -9.816, -3.103], t=2000 -> [-12.616, -3.903, -7.816, -13.016, -12.616, -2.303]
- seed 5: gate forced open every step (2999/2999); online PPL 20.629; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-13.816, -13.816, -11.416, -2.303, -13.416, -13.816]; per-coord range (max-min) [9.2, 10.313, 11.513, 11.513, 11.513, 11.513]; lambda at boundaries t=1000 -> [-11.416, -13.103, -13.816, -13.816, -9.016, -13.816], t=2000 -> [-13.816, -13.816, -13.816, -2.303, -4.216, -13.016]
- seed 6: gate forced open every step (2999/2999); online PPL 21.170; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-13.816, -13.816, -13.816, -13.816, -13.816, -2.303]; per-coord range (max-min) [10.4, 11.513, 11.513, 11.513, 11.513, 11.513]; lambda at boundaries t=1000 -> [-13.816, -11.016, -13.816, -13.816, -13.816, -12.616], t=2000 -> [-13.416, -13.816, -2.703, -13.816, -3.103, -4.303]
- seed 7: gate forced open every step (2999/2999); online PPL 19.887; lambda start [-6.908, -6.908, -6.908, -6.908, -6.908, -6.908] -> end [-13.816, -13.816, -3.503, -12.616, -12.216, -7.016]; per-coord range (max-min) [11.113, 11.2, 11.513, 11.513, 11.513, 11.513]; lambda at boundaries t=1000 -> [-13.816, -13.816, -13.816, -13.816, -2.303, -5.103], t=2000 -> [-6.216, -12.616, -9.016, -3.503, -2.303, -11.016]
