# Measurement index

The [current experiment report](../docs/experiments_v2.md) interprets the final measurements. JSON reports retain per-question outputs, token counts, request events, source fingerprints and failures. Repeats reuse the same questions and do not increase the independent conversation or question count. SQLite response caches remain private and are excluded from git.

## Current protocol

| Directory | Status and scope |
| --- | --- |
| [suite128-primary-observed-v1-20261018](suite128-primary-observed-v1-20261018/suite.json) | All nine planned cells attempted once; all nine failed in the optional unstructured fused arm. Three baseline and six exact arms completed, providing three complete default-path pairs across three conversations (128 turns, 12 evidence-complete QA each). Use [arm-wise tables](suite128-primary-observed-v1-20261018/derived.json) and [paired/sensitivity analysis](suite128-primary-observed-v1-20261018/analysis_views.json); all nine completed default cold arms passed [strict zero-HTTP replay recovery](suite128-primary-observed-v1-20261018/replay_recovery.json). Whole-cell failure status and full denominator remain in `suite.json`. |
| [context128-observed-v1-20261018](context128-observed-v1-20261018/protocol.json) | Completed three fixed-graph cases, 36 unique QA and 72 live answer calls. Graph and keyword reconstruction made zero HTTP requests; [derived results](context128-observed-v1-20261018/derived.json) compare upstream/deduplicated answer contexts. |
| [joint64-heldout-observed-v1-20261018](joint64-heldout-observed-v1-20261018/suite.json) | Unused observed-profile held-out plan; superseded before any live joint outputs by the [structured-policy pilot amendment](../docs/joint_pilot_protocol.md). No measurements in this directory. |

These runs use `observed-tags-v1`: max 2,000 output tokens, repetition penalty 1.05, the same system instruction requesting at most 16 tags, and observation of complete oversized tag lists without clipping. Truncated or malformed responses remain errors. Each independent planned cell is attempted once; conditional timing/QA summaries must retain the failed-cell denominator. Source snapshots and archives identify the measured code.

## Preserved failures and diagnostics

| Directory | Status and interpretation |
| --- | --- |
| [suite128-three-arm-20261018](suite128-three-arm-20261018/suite.json) | Aborted original 1,000-token pilot: one complete cell, one failed cell, seven pending. The first complete pair also crossed an unrelated GPU allocation transition. Its measurements are not pooled with the primary run. |
| [suite128-primary-bounded-v1-20261018](suite128-primary-bounded-v1-20261018/suite.json) | Aborted strict tag-budget run: no complete pairs, one failed cell, eight pending. Exact execution generated a complete 18-item tag list against the hard limit of 16. Its successful baseline remains visible; no failed pair was waived or resumed. |
| [diagnostics/strengthen_cap2000.json](diagnostics/strengthen_cap2000.json) | One isolated reconstruction of the original failed strengthen request; a 2,000-token cap still truncated a repetitive response. Diagnostic only. |
| [diagnostics/rp105-fixed-controls](diagnostics/rp105-fixed-controls/results.json) | Fixed ten-request repetition-penalty diagnostic: nine controls passed, the selected failure prompt exceeded the preregistered 16-tag criterion. Preserved as a failed protocol. |
| [diagnostics/bounded-tags-v1-fixed-controls](diagnostics/bounded-tags-v1-fixed-controls/results.json) | The same ten controls with the common tag instruction passed their fixed criteria. The later hard-budget run nevertheless failed; these diagnostic checks do not establish general reliability. |

The unused [bounded context plan](context128-bounded-v1-20261018/protocol.json) and [bounded joint plan](joint64-heldout-bounded-v1-20261018/suite.json) contain no live results and were superseded before execution.

## Historical and engineering checks

| Evidence | Scope |
| --- | --- |
| [locomo64-final](locomo64-final/summary.json), [locomo64](locomo64/summary.json) | Earlier single-conversation 64-turn results, described in the [historical report](../docs/experiments.md); separate from the primary protocol. |
| [index_benchmark.json](index_benchmark.json) | Real MiniLM/LoCoMo index microbenchmark; measures index work, not live end-to-end inference. |
| [smoke-three-arm-20261018](smoke-three-arm-20261018/summary.json), [smoke8](smoke8/summary.json) | Small integration smokes, not broad quality or performance estimates. |
| [context-plan-locomo64](context-plan-locomo64/manifest.json) | Offline historical graph replay and context-size plan; zero live answer requests. |
| [clean-client-validation-observed.json](clean-client-validation-observed.json) | Clean isolated CPU environment: dependency compatibility and 213 offline tests. Earlier clean-install snapshots are retained alongside it. |
| [final-package-validation.json](final-package-validation.json), [readme-example-validation.json](readme-example-validation.json) | Installed wheel/API and mocked documentation-example validation; no live inference performance claim. |

Raw manifests may include the original machine's paths to document what actually ran. Portable setup and reproduction commands are in the [project README](../README.md).
