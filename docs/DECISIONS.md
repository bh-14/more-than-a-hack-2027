# Decision log

Every value in `configs/params.toml`, every prior in `possession/external/`, and every change to
them, with the evidence behind it. A row says where a value was chosen: an open-data run (command
+ split + result) or "a priori" with the reason. Nothing is chosen on the provided match.

Unless stated otherwise, runs use `scripts/validate_external.py` / `scripts/sweep.py` on the
**tune** split (J03WQQ, J03WOY, J03WR9) with the noise model at seed 0.

| Date | What | Value | Chosen on | Result / reason |
|---|---|---|---|---|
| 2026-10-27 | Label conversion `MIN_CONTROL_S` | 2.0 s | a priori | CHALLENGE.md: a touch without control is not a change of possession. On the 7 open games it turns 22.9 spells per 5 min (mean 7.3 s) into 15.1 (mean 11.0 s), close to DATA.md's published summary of the public window (17, mean 14.2 s), which was used as a sanity check only |
| 2026-10-27 | Noise: ball missing share | 0.41 | a priori | DATA.md, graded frames |
| 2026-10-27 | Noise: other priors (gap length, crowd/air boosts, player dropout, swaps, jitter) | see `noise.py` | a priori | not measured; calibrating on unlabelled statistics of the provided tracking is question 2 for the organisers |
| 2026-10-27 | Noise: ball outside the pitch is lost | 0.9 | a priori | the tactical camera frames the pitch |
| 2026-10-27 | Pipeline v1, all initial values | `params.toml` at first commit | a priori | v1 baseline: spellF1 0.369 / macroF1 0.559 on tune vs 0.219 / 0.400 for baseline_smooth50 on the same degraded tracking |
| 2026-10-27 | `dead_ball.edge_margin`, `edge_score` | 1.0 m, 0.5 | a priori, checked on tune | adding the "lost at the edge" cue moved tune from 0.335 to 0.369 spellF1 (within the 0.04 noise band, kept because it is the camera's behaviour) |
| 2026-10-27 | `decoder.switch_cost`, `to_dead_cost`, `min_possession_frames` | kept 12, 8, 25 | sweep on tune: switch {6, 12, 24} × to_dead {4, 8, 16} × min {25, 50} | flat: the best 9 of 18 combinations span 0.363–0.378 spellF1, inside the noise band. The decoder is not the bottleneck at v1; gains have to come from the evidence (M1 dead ball, M4 shape) |
