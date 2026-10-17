# Held-out 64-turn paired metrics

All four cells were attempted once: two complete and two failed in joint keyword generation. Sixteen cold arms were planned; 13 were attempted, 11 completed, two failed, and three were not reached. All 11 completed arms passed strict zero-HTTP replay.

Each row uses only same-cell reference/candidate pairs. Repeats are averaged within conversation before equal-conversation aggregation; speed uses mean log ratios. Requests/tokens are build-only means on those same pairs. Repeated QA does not increase the independent conversation or question count.

| Candidate | Pairs / conversations / unique QA | Build / total speed | Build HTTP | Input tokens | Output tokens | Token F1 |
| --- | --- | --- | --- | --- | --- | --- |
| optimized | 3 / 2 / 31 | 1.383× / 1.323× | 198.5 → 194.25 | 151,700 → 144,832.75 | 13,636 → 11,439.75 | 0.445 → 0.437 |
| fused | 2 / 1 / 14 | 1.970× / 1.852× | 206 → 127 | 166,671 → 81,758 | 17,199 → 11,056 | 0.547 → 0.579 |
| joint | 2 / 1 / 14 | 2.301× / 2.159× | 206 → 64 | 166,671 → 68,619 | 17,199 → 7,250 | 0.547 → 0.610 |

Fused and joint comparisons cover only conversation sample 4, two repeats of the same 14 questions. The other conversation’s two joint builds completed but QA keyword generation truncated; no live build timing or QA result is recovered for those incomplete arms. Independent live responses and graph decisions differ across policies, so observed reductions are not pure request-fusion or pipeline effects. All four cells meet the prerecorded GPU allocation-stability rule; shared compute load can still change. Model and fixed schemas were prewarmed before the pilot.

The planned 128-turn extension was not executed: its [recorded scheduling decision](../joint128-heldout-structured-v1-20261018/scheduling_decision.json) exceeded the available time after the required 40-minute finalization reserve.
