# Eval baseline

> The 23 questions in `golden.jsonl` were drafted from the parsed manual chunks and
> reviewed by the user against the manual PDFs on 2026-11-25 (no changes needed).
> The numbers below were measured on exactly this set.

Run with `uv run eval` (add `--answers` to print each answer next to the expected one
for spot-checking; `--data-dir` to point at another index).

## 2026-11-26: after #9 (comparison questions) — current baseline

Same models, index and per-question settings as #7 below, plus #9 for questions about
several devices:

- Each device is searched on its own (`RETRIEVE_K` each, product-filtered) and the
  per-device rankings are interleaved, so every device's best chunk is in the top 5.
  Sections are completed per device from a shared budget
  (`CONTEXT_MAX_CHUNKS_COMPARISON = 20`, at least `RETRIEVE_K` per device).
- Each device is searched twice, with the query as asked and with every device name
  taken out, and the two rankings are interleaved. Ranks of each item's expected page in
  its device's own search:

  | Item | Device | With names | Without names |
  | --- | --- | --- | --- |
  | cmp-memory-speed | r16 / act1250 | 1 / 1 | 1 / 1 |
  | cmp-m2-2280-capacity | act1250 / r16 | 1 / 1 | 4 / 2 |
  | cmp-wifi-7 | r16 / act1250 | 1 / 1 | 1 / 2 |
  | cmp-gpu-length | act1250 / r16 | 1 / not in top 6 | 1 / 1 |

  Names pull up each manual's spec chunks (headed "Specifications of Alienware Aurora
  …"), which is usually right but buried the R16's graphics card length on p.61. Neither
  query alone hits every item; interleaved, each first hit is in its device's top 2.
- Context reaches the generator grouped under a header per device, and a comparison
  rule (added to the prompt only for several devices) asks for a bold heading per
  device, per-device citations, and saying under its heading when a device's manual
  doesn't cover the topic.
- "Compare that with the R16" adds the R16 to the session's devices: a message naming
  a device extends the active set when it has a comparison word (compare, versus, vs,
  against, differ…, both), and replaces it otherwise ("what about the R16?").

| Metric | Result |
| --- | --- |
| Device-resolution accuracy | 23/23 (100%) |
| Retrieval page-hit@5 | 16/16 (100%) |
| Correct-refusal rate | 4/4 (100%), false refusals: 0 |

`cmp-gpu-length`, the only failure since #5, now passes. A first version applied the
heading rule to every question; single-device refusals then came back as
"**Alienware Aurora R16** / Not found in the manuals." and were scored as answers
(refusals 2/4), so the rule is now sent only for comparisons, and refusal detection
ignores device-heading lines.

## 2026-11-26: after #7 (answer rules)

Same models, index, `RETRIEVE_K` and refusal as #6 below, plus #7: each retrieved chunk
brings the rest of its section (up to 4 chunks per section, 12 in all) so procedures and
spec tables reach the generator whole; prompt rules for spec options and configuration
variants; a `needs-config-clarification` status; any WARNING/CAUTION of a procedure the
answer gives that the generator left out is appended as "Safety notes from the manual".
The refusal threshold still judges only the ranked hits, not the chunks added to
complete a section.

| Metric | Result |
| --- | --- |
| Device-resolution accuracy | 23/23 (100%) |
| Retrieval page-hit@5 | 15/16 (94%) |
| Correct-refusal rate | 4/4 (100%), false refusals: 0 |

Metrics are unchanged from #6; the only failure is still `cmp-gpu-length`. Page-hit@5
is unaffected by section completion, because added chunks come after the ranked ones.
Every configuration-dependent item came back `answered` with each variant labelled (no
clarification needed). In `cfg-act1250-cpu-cooler-removal` the generator paraphrased the
technicians-only CAUTION and the safety notes restored the manual's wording. This run
also showed safety notes wrongly added to `sd-act1250-bios-key` (BIOS-update cautions,
through a cited p.125-126). The page fallback now applies only to answers with numbered
steps, and a test covers it; the metrics don't depend on it.

## 2026-11-25: after #6 (refusal)

Same models, index and `RETRIEVE_K` as below, plus #6: two refusal layers, both
returning status `refused`.

1. **Retrieval threshold** (`config.REFUSE_BELOW = 0.40`): when no retrieved chunk's
   cosine similarity to the query reaches it, `answer` refuses without calling the
   generator, saying the question may not have been clear.
2. **Prompt-level refusal**: the generator is told to reply exactly "Not found in the
   manuals." when the passages answer no part of the question; a reply opening with
   that is reported as `refused`.

| Metric | Result |
| --- | --- |
| Device-resolution accuracy | 23/23 (100%) |
| Retrieval page-hit@5 | 15/16 (94%) |
| Correct-refusal rate | 4/4 (100%), false refusals: 0 |

The only failure is `cmp-gpu-length` (comparison retrieval, unchanged; see below).

### Why the threshold is 0.40

Best-chunk cosine similarity overlaps between the two groups, so no threshold separates
answerable from not-in-manual questions:

| Question | Best similarity | Manuals answer it? |
| --- | --- | --- |
| Which holds its resale value better, the R16 or the ACT1250? | 0.499 | no |
| How do I install Ubuntu on the ACT1250? | 0.535 | no |
| How loud is the Aurora R16 in decibels under full load? | 0.689 | no |
| What is the retail price of the Alienware Aurora ACT1250? | 0.731 | no |
| CR2032 (bare part number) | 0.406 | yes |
| What does the CR2032 do? | 0.438 | yes |
| What is the CR2032 battery used for? | 0.522 | yes |
| Which supports Wi-Fi 7, the R16 or the ACT1250? | 0.566 | yes |
| Every other answerable eval and test question | 0.572 – 0.842 | yes |

Short questions built around a part number or model code score low even when the right
passage is retrieved, so any threshold that refused resale or Ubuntu would also refuse
real questions. With the threshold off, the prompt-level refusal alone refused all four
not-in-manual questions and nothing else. The threshold is therefore a safety floor just
below the lowest answerable question measured: it skips generation only for queries
that match nothing at all, and catches none of the eval set today. Removing device
names from the query before search was also measured: it lifted some short questions
(0.47 → 0.6) but dropped the Wi-Fi 7 comparison to 0.453, so it was not adopted.

Test aliases now use realistic names (`1250`, `r-16`, and `aurora` for both devices)
in place of made-up nicknames, which scored low only because the embedder can't place
them.

## 2026-11-25: after #8 (conversational session)

Same models, index and `RETRIEVE_K` as below, plus #8: questions with first-person
words ("my", "I", "mine") are rewritten into a standalone query before retrieval.

| Metric | Result |
| --- | --- |
| Device-resolution accuracy | 23/23 (100%) |
| Retrieval page-hit@5 | 15/16 (94%) |
| Correct-refusal rate | 0/4 (0%), false refusals: 0 |

Failures:

- `cmp-gpu-length` (comparison): got ACT1250 pages 23, 26, 27, 61 and no R16 chunk in
  the top 5. Comparison retrieval still isn't balanced per device.
- `nim-*` (all four not-in-manual questions): status `answered`, expected `refused`.
  Still 0% by construction until a `refused` status exists.

`cfg-act1250-cpu-cooler-removal` now passes: both the air-cooler and the liquid-cooler
removal pages are in the top 5. The question says "my", so it goes through the #8
rewrite. One run can't separate that from run-to-run variation in retrieval.

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
