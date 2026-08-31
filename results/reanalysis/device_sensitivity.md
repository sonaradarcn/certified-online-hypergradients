# B5. Device-stratified sensitivity (GPU `results/e2` vs CPU `results/e2_controls`)

Same code, same seeds, same config (`mackey_drift`, lr0=0.003, 12000 steps, gamma 0.9, alpha 0.4, K 10, rank 4, c=2, M_H=5, no fail-closed); the ONLY difference is the device, i.e. a float-reassociation perturbation of order 1e-7 relative.  `regime` marks a run unstable if events > 30 or NMSE > 1.  The CPU counterpart of the GPU `cohg` arm is the control study's reference arm `..._mh5_fc0_s{S}.json`, and of `cohg_nogate` it is `..._nogate_lr0.003_a0.4_s{S}.json`; the config fields inside those files were checked to match the GPU runs.

`open coord-steps` = `coord_open_frac` x 12000 steps x 6 LR groups, i.e. the raw count of gate-open decisions.  `1st diff step` is the first index at which the two per-step loss traces are not bitwise equal; `max rel traj` is the largest relative gap over the whole trace; `d ln-LR` is the max-norm gap between the two final per-group NATURAL-log LR vectors (0.69 = a 2x LR difference).

## certificate-gated COHG (alpha=0.4)  (`cohg`)

### outcome metrics

| seed | GPU NMSE | CPU NMSE | \|dNMSE\| | rel diff | GPU events | CPU events | \|d ev\| | regime flip |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0148825 | 0.0148825 | 3.185e-09 | 2.14e-07 | 0 | 0 | 0 | no |
| 1 | 0.0184538 | 0.0184537 | 6.373e-09 | 3.45e-07 | 0 | 0 | 0 | no |
| 2 | 0.0172826 | 0.0172826 | 2.122e-09 | 1.23e-07 | 0 | 0 | 0 | no |
| 3 | 0.0185398 | 0.0185398 | 5.266e-09 | 2.84e-07 | 2 | 2 | 0 | no |
| 4 | 0.0145733 | 0.0145733 | 6.823e-09 | 4.68e-07 | 7 | 7 | 0 | no |
| 5 | 0.0126028 | 0.0148538 | 2.251e-03 | 1.52e-01 | 0 | 0 | 0 | no |
| 6 | 0.0150296 | 0.0191812 | 4.152e-03 | 2.16e-01 | 0 | 0 | 0 | no |
| 7 | 0.0126048 | 0.0149288 | 2.324e-03 | 1.56e-01 | 28 | 28 | 0 | no |
| 8 | 0.0126803 | 0.0126813 | 9.538e-07 | 7.52e-05 | 28 | 28 | 0 | no |
| 9 | 0.0136986 | 0.0162798 | 2.581e-03 | 1.59e-01 | 25 | 25 | 0 | no |

### gate decisions, cost, and where the perturbation enters

| seed | GPU open coord-steps | CPU open coord-steps | d open | GPU open rate | CPU open rate | 1st diff step | max rel traj | d ln-LR | GPU HVPs | CPU HVPs | GPU wall (s) | CPU wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 33 | 33 | +0 | 4.583e-04 | 4.583e-04 | 0 | 1.66e-06 | 0.000 | 94588 | 94588 | 3741 | 5041 |
| 1 | 26 | 26 | +0 | 3.611e-04 | 3.611e-04 | 1 | 2.25e-06 | 0.000 | 94588 | 94588 | 3739 | 5160 |
| 2 | 45 | 45 | +0 | 6.250e-04 | 6.250e-04 | 0 | 1.67e-06 | 0.000 | 94588 | 94588 | 3805 | 5171 |
| 3 | 25 | 25 | +0 | 3.472e-04 | 3.472e-04 | 4 | 1.94e-06 | 0.000 | 94588 | 94588 | 3920 | 5019 |
| 4 | 33 | 33 | +0 | 4.583e-04 | 4.583e-04 | 0 | 2.69e-06 | 0.000 | 94588 | 94588 | 3842 | 5175 |
| 5 | 36 | 31 | +5 | 5.000e-04 | 4.306e-04 | 0 | 5.90e-01 | 0.524 | 94588 | 94588 | 3524 | 5175 |
| 6 | 42 | 36 | +6 | 5.833e-04 | 5.000e-04 | 0 | 7.04e-01 | 0.596 | 94588 | 94588 | 3829 | 5013 |
| 7 | 37 | 32 | +5 | 5.139e-04 | 4.444e-04 | 4 | 5.82e-01 | 0.518 | 94588 | 94588 | 3673 | 5044 |
| 8 | 51 | 51 | +0 | 7.083e-04 | 7.083e-04 | 0 | 3.09e-03 | 0.000 | 94588 | 94588 | 2813 | 5048 |
| 9 | 34 | 28 | +6 | 4.722e-04 | 3.889e-04 | 0 | 6.35e-01 | 0.533 | 94588 | 94588 | 3683 | 5170 |

- mean paired |dNMSE| = 1.131e-03 (median relative 3.78e-05, max relative 2.16e-01)
- mean paired |d events| = 0.00 (max 0)
- arm mean NMSE: GPU 0.0150 vs CPU 0.0162; arm MEDIAN NMSE: GPU 0.0147 vs CPU 0.0156
- seeds with a BITWISE-IDENTICAL loss trajectory: **0 / 10**; among the rest the traces first differ at step 0-4 (median 0)
- seeds that change regime (events>30 OR NMSE>1): **0 / 10**
- seeds that change the stricter divergence flag (NMSE>1 alone): **0 / 10**
- seeds whose realized per-coordinate GATE-OPEN RATE differs between devices: **4 / 10**

## same estimator, gate OFF (pure sign, alpha=0.4)  (`cohg_nogate`)

### outcome metrics

| seed | GPU NMSE | CPU NMSE | \|dNMSE\| | rel diff | GPU events | CPU events | \|d ev\| | regime flip |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.00150854 | 0.00150854 | 1.826e-10 | 1.21e-07 | 33 | 33 | 0 | no |
| 1 | 0.00293788 | 0.00293788 | 4.694e-10 | 1.60e-07 | 111 | 111 | 0 | no |
| 2 | 0.0014704 | 0.0014704 | 1.422e-10 | 9.67e-08 | 91 | 91 | 0 | no |
| 3 | 0.00272676 | 0.00272676 | 3.137e-11 | 1.15e-08 | 178 | 178 | 0 | no |
| 4 | 0.00321441 | 0.00321673 | 2.325e-06 | 7.23e-04 | 107 | 108 | 1 | no |
| 5 | 59.8231 | 0.00478535 | 5.982e+01 | 1.00e+00 | 298 | 53 | 245 | no |
| 6 | 0.00299335 | 0.00299335 | 1.064e-10 | 3.55e-08 | 64 | 64 | 0 | no |
| 7 | 0.00445068 | 0.00445068 | 3.763e-09 | 8.45e-07 | 97 | 97 | 0 | no |
| 8 | 0.00301476 | 0.00325758 | 2.428e-04 | 7.45e-02 | 163 | 168 | 5 | no |
| 9 | 0.00336367 | 0.00336367 | 1.439e-10 | 4.28e-08 | 147 | 147 | 0 | no |

### gate decisions, cost, and where the perturbation enters

| seed | GPU open coord-steps | CPU open coord-steps | d open | GPU open rate | CPU open rate | 1st diff step | max rel traj | d ln-LR | GPU HVPs | CPU HVPs | GPU wall (s) | CPU wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 0 | 1.26e-04 | 0.000 | 11992 | 11992 | 522 | 633 |
| 1 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 1 | 6.01e-05 | 0.000 | 11992 | 11992 | 556 | 634 |
| 2 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 0 | 1.93e-04 | 0.000 | 11992 | 11992 | 527 | 639 |
| 3 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 3 | 5.10e-04 | 0.000 | 11992 | 11992 | 546 | 638 |
| 4 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 0 | 5.97e-01 | 3.913 | 11992 | 11992 | 553 | 704 |
| 5 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 0 | 1.00e+00 | 11.513 | 11992 | 11992 | 576 | 710 |
| 6 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 0 | 2.78e-05 | 0.000 | 11992 | 11992 | 567 | 646 |
| 7 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 3 | 3.30e-05 | 0.000 | 11992 | 11992 | 579 | 637 |
| 8 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 0 | 9.78e-01 | 3.200 | 11992 | 11992 | 340 | 778 |
| 9 | gate off | gate off | gate off | 1.000 (open) | 1.000 (open) | 0 | 1.26e-05 | 0.000 | 11992 | 11992 | 573 | 636 |

- mean paired |dNMSE| = 5.982e+00 (median relative 1.40e-07, max relative 1.00e+00)
- mean paired |d events| = 25.10 (max 245)
- arm mean NMSE: GPU 5.9849 vs CPU 0.0031; arm MEDIAN NMSE: GPU 0.0030 vs CPU 0.0031
- seeds with a BITWISE-IDENTICAL loss trajectory: **0 / 10**; among the rest the traces first differ at step 0-3 (median 0)
- seeds that change regime (events>30 OR NMSE>1): **0 / 10**
- seeds that change the stricter divergence flag (NMSE>1 alone): **1 / 10**
- seeds whose realized per-coordinate GATE-OPEN RATE differs between devices: **0 / 10**

## Summary

| arm | n | mean \|dNMSE\| | median rel | max rel | mean \|d ev\| | bitwise-identical traj | regime flips (ev>30 or NMSE>1) | divergence flips (NMSE>1) | gate-rate differs | GPU mean | CPU mean | GPU median | CPU median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cohg | 10 | 1.131e-03 | 3.78e-05 | 2.16e-01 | 0.00 | 0/10 | 0 | 0 | 4 | 0.0150 | 0.0162 | 0.0147 | 0.0156 |
| cohg_nogate | 10 | 5.982e+00 | 1.40e-07 | 1.00e+00 | 25.10 | 0/10 | 0 | 1 | 0 | 5.9849 | 0.0031 | 0.0030 | 0.0031 |

## Reading

**The gated arm never changes regime, but it is not bit-identical across devices.** Six of ten COHG seeds reproduce to fp32 round-off (rel <= 1e-4); the other four (5, 6, 7, 9) differ by 15-22% in NMSE because the float perturbation moves a handful of borderline `|ghat_j|` values across the `c * beta_j` threshold, so the realized gate-open rate changes -- and those are exactly the four seeds whose open coordinate-step COUNT differs (seed 5: 36 GPU vs 31 CPU; seed 6: 42 vs 36; seed 7: 37 vs 32; seed 9: 34 vs 28). The instability-event count is nevertheless IDENTICAL on all ten seeds (mean paired |d events| = 0), and both arm summaries land in the same place (GPU 0.0150 vs CPU 0.0162, i.e. within half a pooled SD of 0.002). So the certificate gate's operating point is device-stable; its exact NMSE to three digits is not, and the paper should not quote COHG's NMSE to more than two significant figures on the strength of a single device.  (This refines the claim in `results/e2_controls/SUMMARY.md` that the two devices agree 'with identical gate decisions': that holds for the single config that was spot-checked, not for every seed.)

**The perturbation enters immediately; the gate decides whether it matters.** No pair is bitwise identical from step 0 -- every one of the twenty pairs separates within the first handful of steps, which is what a reassociation-only difference should do. What differs between arms is the AMPLIFICATION. In the gated arm the divergence stays bounded: the four gate-flipping seeds end 0.52-0.60 nats apart in the worst LR group (a 1.7-1.8x LR difference) and stay inside the same NMSE decade, and the other six end bit-for-bit on the same lambda. In the ungated arm the same perturbation is amplified without limit: seeds 4, 8 and 5 end 3.9, 3.2 and 11.5 nats apart (the last is a 1e5x LR difference), and seed 5 runs away.

**The ungated arm changes regime on a float.** Seed 5 lands at NMSE 59.8 with 298 events on GPU and at NMSE 0.0048 with 53 events on CPU: a ~1e-7 reassociation decides which side of the divergence boundary the run falls on, moving that seed's NMSE by four orders of magnitude and the ARM MEAN by a factor of ~2000, while the arm MEDIAN barely moves (0.0030 vs 0.0031). Under the paper's coarse regime label the seed is 'unstable' on both devices (298 and 53 events both exceed 30), so the headline regime count is unchanged (0/10 flips) -- but under the strict NMSE>1 divergence flag it flips 1/10. Two further ungated seeds (4 and 8) also disagree in event count (107 vs 108, 163 vs 168) without changing regime.

**Cost is a device artefact, not a method property.** The gated arm issues the same 94588 HVPs on both devices and the ungated arm the same 11992, so nothing about the algorithm's work changes; only wall time does (GPU ~3.7ks vs CPU ~5.1ks for COHG), because at 13k parameters the HVPs are launch-bound and a CPU core beats a 3080. Device choice therefore cannot be read off the timing column as an efficiency claim either way.

**Consequences.** (i) Report medians and event counts alongside means for the ungated controls -- the mean of an arm that straddles a divergence boundary is not a stable statistic. (ii) Keep every within-study comparison on one device, as results/e2_controls does. (iii) Quote COHG NMSE to two significant figures. (iv) The gate is what removes the device sensitivity of the OUTCOME, which is the same tail-control claim the paper makes on other grounds -- it does not make the run reproducible bit-for-bit, and the paper should not claim that.
