# Eval baseline

> **DRAFT eval set.** The 23 questions in `golden.jsonl` were drafted from the parsed
> manual chunks and have not yet been reviewed by the user (every item has
> `"reviewed": false`). Re-run and replace these numbers once the set is verified.

Run with `uv run eval` (add `--answers` to print each answer next to the expected one
for spot-checking; `--data-dir` to point at another index).

## 2026-11-25: first baseline

Pipeline as of #4 (device registry + resolution), on top of the #3 tracer bullet:
hybrid search, `RETRIEVE_K = 6`, generator `frob/qwen3.5-instruct:9b`, embedder
`qwen3-embedding:0.6b`. No refusal status, no comparison balancing, no reranker.
Index: both manuals (ACT1250, R16), built by `index`.

| Metric | Result |
| --- | --- |
| Device-resolution accuracy | 23/23 (100%) |
| Retrieval page-hit@5 | 14/16 (88%) |
| Correct-refusal rate | 0/4 (0%), false refusals: 0 |

Page-hit@5 counts an item as a hit only when, for every expected device, every expected
page group has a page in one of the top 5 retrieved chunks from that device's manual.
Configuration-dependent procedures have one group per configuration, so finding only
one variant counts as a miss.

Failures:

- `cmp-gpu-length` (comparison): the top 5 chunks were all from the ACT1250 manual, so
  the R16's graphics-card dimensions (p.61) were never retrieved and the answer said the
  R16 manual doesn't cover it. Comparison retrieval isn't balanced per device yet.
- `cfg-act1250-cpu-cooler-removal` (configuration-dependent): the liquid-cooler removal
  (p.83-84) was retrieved but the air-cooler removal (p.81-82) was not, so the answer
  can't present both configurations.
- `nim-*` (all four not-in-manual questions): status `answered`, expected `refused`.
  The pipeline has no `refused` status yet, so this rate is 0% by construction. The
  answer text did say the information isn't in the manuals in all four cases (prose,
  not graded).
