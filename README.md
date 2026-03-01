# microgpt-go

The most atomic way to train and run inference for a GPT in pure, dependency-free Go.

This is a Go translation of [@karpathy](https://github.com/karpathy)'s [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a minimal GPT implementation that includes everything from autograd to training to inference in a single file with zero external dependencies.

## What's Inside

- **Autograd Engine** — A `Value` type that tracks computation graphs and computes gradients via backpropagation
- **GPT Model** — Token/position embeddings, multi-head attention, RMSNorm, and MLP blocks (follows GPT-2 architecture with minor simplifications)
- **Adam Optimizer** — With linear learning rate decay
- **Training Loop** — Trains on any line-delimited text dataset
- **Model Persistence** — Save and load trained weights to/from JSON
- **Inference** — Temperature-controlled text generation via CLI or HTTP server
- **Streaming Training** — Train on data piped from stdin for evolutionary/incremental workflows

## Quick Start

```bash
go build -o microgpt
./microgpt
```

On the first run, the program will automatically download the [names dataset](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt) to `input.txt`. Training runs for 1000 steps, then generates 20 new hallucinated names.

## CLI Reference

```bash
# Train on a custom dataset
./microgpt -dataset cities.txt -steps 2000

# Train from stdin (streaming / evolutionary training)
cat my_corpus.txt | ./microgpt -dataset - -save model.json

# Save trained model to disk
./microgpt -save model.json

# Load a saved model and generate samples
./microgpt -load model.json -mode infer -samples 10 -temp 0.7

# Start an HTTP inference server
./microgpt -load model.json -mode serve -addr :8080

# Combine options
./microgpt -dataset recipes.txt -steps 500 -save model.json
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-dataset` | `input.txt` | Path to training data, or a well-known dataset name (see `-list-datasets`), or `-` for stdin |
| `-steps` | `1000` | Number of training steps |
| `-temp` | `0.5` | Sampling temperature (0, 1] |
| `-samples` | `20` | Number of samples to generate |
| `-save` | | Save trained weights to this JSON file |
| `-load` | | Load weights from this JSON file (skip training) |
| `-mode` | `train` | `train` (train + infer), `infer` (generate only), or `serve` (HTTP server) |
| `-addr` | `:8080` | Address for the HTTP server (used with `-mode serve`) |
| `-list-datasets` | | List available well-known datasets and exit |

## Well-Known Datasets

microgpt-go ships with a registry of well-known datasets that can be referenced by
name. When you pass a well-known name to `-dataset`, the file is automatically
downloaded on first use and cached in `~/.cache/microgpt-go/` so subsequent runs
skip the download.

```bash
# List all available datasets
./microgpt -list-datasets

# Train on the built-in names dataset (cached after first download)
./microgpt -dataset names -steps 1000

# Train on English dictionary words
./microgpt -dataset words -steps 2000 -save word_model.json
```

| Name | Category | Description |
|------|----------|-------------|
| `names` | names | 32K human first names (from Karpathy's makemore) |
| `words` | vocabulary | 370K English dictionary words |

The default `-dataset input.txt` behaviour is unchanged — if `input.txt` doesn't
exist, the names dataset is downloaded to it directly (no caching).

## HTTP Inference Server

When running with `-mode serve`, the server exposes:

```
POST /generate
Content-Type: application/json

{"prompt": "mar", "temperature": 0.5, "max_tokens": 16}
```

Response:

```json
{"text": "maria"}
```

## Streaming Evolutionary Training

You can pipe data from any source into microgpt-go for incremental training workflows:

```bash
# Train on live data from a stream
tail -f /var/log/access.log | awk '{print $7}' | ./microgpt -dataset - -steps 500 -save url_model.json

# Chain training sessions (evolutionary)
./microgpt -dataset batch1.txt -steps 500 -save model.json
./microgpt -load model.json -dataset batch2.txt -steps 500 -save model.json
```

## Running Tests

```bash
go test -v ./...
```

## Architecture

The model is a single-layer transformer with:
- 16-dimensional embeddings
- 4 attention heads
- 16-token context window
- RMSNorm (instead of LayerNorm)
- ReLU activation (instead of GeLU)
- No biases

## Use Cases and Evolution

For a detailed discussion of practical use cases, real-world datasets, and ideas for evolving microgpt-go, see [Use Cases and Evolution](docs/USE_CASES_AND_EVOLUTION.md).

## Microservices Integration

For patterns on periodic API ingestion, per-service models, and integration with platforms like [Mu](https://github.com/micro/mu), see [Microservices Integration](docs/MICROSERVICES.md).

## AI Assistant Guide

See [CLAUDE.md](CLAUDE.md) for project guidance when working with AI coding assistants.

## Credits

Based on the original Python implementation by [Andrej Karpathy](https://github.com/karpathy).
