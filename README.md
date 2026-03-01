# microgpt-go

The most atomic way to train and run inference for a GPT in pure, dependency-free Go.

This is a Go translation of [@karpathy](https://github.com/karpathy)'s [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a minimal GPT implementation that includes everything from autograd to training to inference in a single file with zero external dependencies.

## What's Inside

- **Autograd Engine** — A `Value` type that tracks computation graphs and computes gradients via backpropagation
- **GPT Model** — Token/position embeddings, multi-head attention, RMSNorm, and MLP blocks (follows GPT-2 architecture with minor simplifications)
- **Adam Optimizer** — With linear learning rate decay
- **Training Loop** — Trains on a character-level names dataset
- **Inference** — Temperature-controlled text generation

## Usage

```bash
go build -o microgpt
./microgpt
```

On the first run, the program will automatically download the [names dataset](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt) to `input.txt`. Training runs for 1000 steps, then generates 20 new hallucinated names.

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

## Credits

Based on the original Python implementation by [Andrej Karpathy](https://github.com/karpathy).
