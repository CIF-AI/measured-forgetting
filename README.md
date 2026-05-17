# Measured Forgetting

Context management for agentic local LLMs on consumer hardware.

**Paper:** [Measured Forgetting: Context Management for Agentic Local LLMs](https://arxiv.org/abs/TODO) (arXiv, May 2026)

## What is this?

When a local LLM (8B–14B parameters, 8K–16K context) acts as a tool-calling agent, it exhausts its context window in 3–5 rounds. Measured forgetting treats context as a finite resource: it monitors utilisation in real-time, partitions the message history into three zones (pinned, compressible, recent), and invokes the model itself to compress the compressible zone — all within the same generation loop.

**Version 2** extends uniform compression with three instruments:
- **Influence Equation** — multi-dimensional scoring with super-linear Φ(B) = B^1.5 resonance
- **Trace Topology** — causal chain preservation (binding units compressed together)
- **Problem Taxonomy** — 6-class adaptive compression strategy (lookup, multi-hop, exploratory, aggregation, contradiction, temporal)

## Results

Across 7 models from 5 independent families, V2 outperforms V1 on every model tested (sign test: p = 0.008):

| Model | Family | Baseline | V1 | V2 | V2−V1 |
|-------|--------|----------|------|------|-------|
| Qwen3 8B | Alibaba | 0.50 | 0.35 | **0.81** | +0.46 |
| Phi-4 mini 3.8B | Microsoft | 2.02 | 2.58 | **2.98** | +0.40 |
| Llama 3.2 3B | Meta | 1.67 | 2.69 | **2.90** | +0.21 |
| Mistral 7B | Mistral | 2.06 | 2.69 | **2.85** | +0.17 |
| Qwen2.5 7B | Alibaba | 1.90 | 2.79 | **2.94** | +0.15 |
| Gemma 3 4B | Google | 1.83 | 2.69 | **2.83** | +0.15 |
| Mistral-Nemo 12B | Mistral | 1.85 | 2.71 | **2.85** | +0.15 |

## Repository structure

```
src/
  lib.rs                    — crate root (Message type)
  forgetting.rs             — measured forgetting v2 algorithm
  benchmark.rs              — 18 scenarios, 47 probes, scoring
  bin/bench_forgetting.rs   — CLI runner (Ollama API)
benchmark_results/          — JSON results for all 7 models
paper/                      — LaTeX source
```

## Running the benchmark

Requires [Ollama](https://ollama.ai) with models pulled locally.

```bash
# Single model
cargo run --release --bin bench-forgetting -- --model mistral:7b

# JSON output
cargo run --release --bin bench-forgetting -- --model mistral:7b --json > results.json

# All default models
cargo run --release --bin bench-forgetting -- --all
```

## Production deployment

The algorithm runs in production in [B.app](https://thexi.dev), a Tauri desktop application using Qwen3 8B on Apple M4. The production integration (KV cache persistence, session worker, tool-calling orchestrator) is not included in this repo — this contains only the compression algorithm and benchmark.

## Citation

```bibtex
@article{asidi2026measured,
  title={Measured Forgetting: Context Management for Agentic Local LLMs},
  author={Asidi, Barak Achillah},
  journal={arXiv preprint},
  year={2026}
}
```

## License

Apache 2.0

## Contact

We welcome adversarial benchmarks, alternative Φ formulations, and replications on different hardware tiers.

**research@thexi.dev** · [thexi.dev](https://thexi.dev) · [CIF AI](https://github.com/CIF-AI)
