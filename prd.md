# PRD: Alienware Manuals RAG

## Goal
A fully local retrieval-augmented generation (RAG) system that answers questions about Alienware desktop and laptop manuals stored in `Manuals/`. It is a portfolio/learning project that is also used personally: simple to run, with a small evaluation set that demonstrates it works.

## Scope
- Corpus: Alienware desktops and laptops only. Grows over time by dropping PDFs into `Manuals/` and re-running `index` (incremental — unchanged files are not re-embedded).
- Initial manuals: Alienware Aurora ACT1250, Alienware Aurora R16.
- Everything runs locally on an RTX 5070 Ti (16 GB VRAM). No cloud APIs.

## Stack
| Concern | Choice |
|---|---|
| Language / tooling | Python 3.14, uv, `pyproject.toml` + lockfile |
| Entry points | `index` (build/update the index), `ask` (chat CLI) |
| Parsing | Docling — OCR off by default (verify PDFs have a text layer), table structure on |
| Chunking | Docling `HybridChunker`, token-limited to the embedder's tokenizer; heading path prepended before embedding |
| Chunk metadata | `product`, `page`, `section heading path`, linked image paths |
| Images | Exported and linked to page/chunk. VLM captions (`gemma4:12b`) are an optional, toggleable ingest stage, kept only if eval shows a gain |
| Embedder | `qwen3-embedding:0.6b` via Ollama (fallback upgrade: `qwen3-embedding:4b`) |
| Vector store | LanceDB, hybrid search (BM25 + vector). Reranker only if eval demands it |
| Generator | `frob/qwen3.5-instruct` (9b tag) via Ollama, 16k context; model name configurable |
| Framework | Hand-rolled (`ollama` client + LanceDB), no LangChain/LlamaIndex |
| UI | CLI first, then a Gradio front-end |
| Git | `Manuals/` and `data/` (index, exported images) are gitignored |

## Query pipeline
1. **Device resolution (first step).** Determine which device(s) the question is about — may be several for comparison questions.
   - Device registry generated at ingest from filename (model code, e.g. `act1250`) and title page (marketing name, e.g. "Alienware Aurora ACT1250"), plus a hand-editable alias file. Both code and name are aliases; answers display the marketing name.
   - Deterministic alias matching first; LLM fallback (structured JSON output) only when there is no match or the match is ambiguous.
   - No device detected: follow-ups inherit the session's device set; otherwise ask "which device?" listing known devices.
2. **Conversation memory.** Session keeps the active device set, the user's stated configuration (e.g. "5080, liquid-cooled"), and the last few turns. Follow-ups are rewritten into standalone queries before retrieval. Nothing persists across sessions in v1.
3. **Retrieval.** Per-device retrieval (top-k each), filtered by `product` metadata, context grouped under device headers so one manual cannot crowd out another.
4. **Refusal.** Two layers: (a) if the best retrieval score is below a threshold (tuned on the eval set), answer "not found in the manuals" without calling the LLM; (b) the prompt instructs the model to refuse when the retrieved context does not contain the answer. Never answer from general knowledge.
5. **Generation rules.**
   - Every claim cites manual + page.
   - Procedures are returned as numbered steps in manual order.
   - Any WARNING/CAUTION attached to a procedure (ESD, battery, power-off) is always included.
   - Spec questions list every configuration option the manual documents (e.g. ACT1250 GPU: 5070 / 5070 Ti / 5080), never a single guessed value.
   - Configuration-dependent procedures: present each variant labeled by configuration; ask a clarifying question only when variants are too long to present together. Use the user's stated configuration if known.
   - Comparison answers state explicitly when one manual does not cover the topic.

## Configuration variance
The same model ships in multiple configurations (GPU, cooling, storage, PSU). Manuals describe per-spec differences inline, so v1 does **not** tag chunks by configuration variant; it relies on the manual's text plus the model. The eval set includes config-dependent questions to detect if this proves insufficient.

## Evaluation
- ~20 golden questions, drafted by Claude from the parsed manuals and verified by the user. Each has expected device(s), expected page(s), and the expected answer.
- Coverage: single-device, comparison, no-device-mentioned, config-dependent, and not-in-manual (should refuse).
- Automated metrics: device-resolution accuracy, retrieval page-hit@5, correct-refusal rate.
- Answer quality: spot-checked manually (no LLM judge).

## Build order
Each stage is validated before the next.
1. Docling parse and inspect chunks and exported images
2. Device registry (codes, marketing names, aliases)
3. LanceDB hybrid retrieval, plus the eval set and device/page-hit metrics
4. Generation and the `ask` CLI (citations, refusal, warnings)
5. Conversation memory and comparison retrieval
6. VLM image-caption experiment (toggle, measured by eval)
7. Gradio UI with cited page/image display

## Project conventions
- Commits and PRs contain no Claude co-author trailers or attribution.
