# CLAUDE.md

Project guidance for AI assistants working with this codebase.

## What is this?

microgpt-go is a minimal GPT implementation in pure Go with zero external dependencies. It includes autograd, training, model persistence, an HTTP inference server, and a CLI — all in a single file. Translated from Andrej Karpathy's microgpt.py.

## Build and run

```bash
go build -o microgpt && ./microgpt
```

## Run tests

```bash
go test -v ./...
```

## CLI modes

```bash
# Train on default names dataset (downloads automatically)
./microgpt

# Train on a custom dataset file
./microgpt -dataset mydata.txt

# Train from stdin (streaming)
echo -e "hello\nworld" | ./microgpt -dataset -

# Save trained model to disk
./microgpt -save model.json

# Load a saved model and generate samples
./microgpt -load model.json -mode infer

# Start an HTTP inference server
./microgpt -load model.json -mode serve -addr :8080

# Configure training
./microgpt -steps 2000 -temp 0.7 -samples 10

# Combine options
./microgpt -dataset mydata.txt -steps 500 -save model.json
```

## Architecture

- `Value` type with computation graph for automatic differentiation
- Single-layer transformer: 16-dim embeddings, 4 attention heads, 16-token context
- RMSNorm, ReLU activation, no biases
- Adam optimizer with linear LR decay
- Character-level tokenizer

## Key design decisions

- Zero external dependencies — only Go stdlib
- Everything in one file for readability and portability
- Scalar autograd (not tensor) — educational clarity over performance
- Model weights serialize to JSON for easy inspection and interop

## What NOT to do

- Don't add external dependencies — the zero-dependency constraint is intentional
- Don't split into multiple packages — single-file design is a feature
- Don't replace scalar autograd with tensors — that changes the project's purpose
