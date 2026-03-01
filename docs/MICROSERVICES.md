# Microservices Integration

This document describes how microgpt-go can be used as a lightweight model layer inside microservice architectures — including periodic API ingestion, per-service models, and integration with platforms like [Mu](https://github.com/micro/mu).

## Overview

microgpt-go already supports three capabilities that map directly to microservice patterns:

1. **HTTP inference** (`-mode serve`) — serve predictions over a REST API
2. **Model persistence** (`-save` / `-load`) — swap models at runtime without code changes
3. **Streaming training** (`-dataset -`) — ingest data from any upstream source via stdin

Combining these with a scheduler and an API data source turns microgpt-go into a continuously learning microservice.

## Periodic API Ingestion

Many services expose data as line-oriented text — RSS headlines, log streams, ticker symbols, place names. A cron job or lightweight loop can fetch data from an API, feed it to microgpt-go for incremental training, and replace the model file the inference server reads.

### Pattern

```
 ┌─────────┐     fetch      ┌────────────┐    stdin     ┌───────────┐
 │  API /  │ ──────────────▶│  transform │ ──────────▶│  microgpt  │
 │  Feed   │   (cron/loop)  │  (jq/awk)  │  -dataset - │  -save m.json
 └─────────┘                └────────────┘             └───────────┘
                                                             │
                                                        load model
                                                             │
                                                       ┌───────────┐
                                                       │  microgpt  │
                                                       │ -mode serve│
                                                       └───────────┘
```

### Example: News Headlines

Fetch headlines from an RSS-to-JSON API, extract titles, and train a headline generator that refreshes every hour:

```bash
#!/usr/bin/env bash
# train_news.sh — run via cron every hour

NEWS_URL="https://api.example.com/news/headlines?format=text"
MODEL="news_model.json"

# Fetch fresh headlines (one per line)
curl -sf "$NEWS_URL" > /tmp/headlines.txt

# Incremental training on new data
if [ -f "$MODEL" ]; then
  ./microgpt -load "$MODEL" -dataset /tmp/headlines.txt -steps 200 -save "$MODEL"
else
  ./microgpt -dataset /tmp/headlines.txt -steps 1000 -save "$MODEL"
fi
```

A separate process serves the model:

```bash
./microgpt -load news_model.json -mode serve -addr :9001
```

The inference server does not need to restart — on the next request it can reload the updated JSON file. For zero-downtime swaps, write to a temporary file and rename atomically:

```bash
./microgpt -load "$MODEL" -dataset /tmp/headlines.txt -steps 200 -save "${MODEL}.tmp"
mv "${MODEL}.tmp" "$MODEL"
```

### Example: Weather Descriptions

```bash
# Fetch short weather descriptions for training
curl -sf "https://api.example.com/weather/descriptions" \
  | ./microgpt -dataset - -steps 300 -save weather_model.json
```

### Example: Place Names

```bash
# Periodically learn new place names
curl -sf "https://api.example.com/places?limit=500&format=lines" \
  | ./microgpt -dataset - -steps 200 -save places_model.json
```

## Per-Service Model Architecture

In a microservice platform each service owns a domain-specific dataset. microgpt-go is small enough (single binary, no GPU) to run one instance per service, each with its own model file and inference port.

```
┌─────────────────────────────────────────────┐
│               Service Mesh / Gateway        │
├──────────┬──────────┬───────────┬───────────┤
│  :9001   │  :9002   │  :9003    │  :9004    │
│  News    │  Weather │  Places   │  Chat     │
│  Model   │  Model   │  Model    │  Model    │
│          │          │           │           │
│ headlines│ forecasts│ city names│ responses │
│ .json    │ .json    │ .json     │ .json     │
└──────────┴──────────┴───────────┴───────────┘
      │          │          │           │
      └──────────┴──────────┴───────────┘
              Each service runs:
        microgpt -load <model> -mode serve -addr :<port>
```

Each service independently:

- **Fetches** data from its upstream API or feed
- **Trains** incrementally on new data (scheduled or event-driven)
- **Serves** predictions via `/generate`

This keeps models isolated — a bad training batch in one service does not affect others.

### Docker Compose Example

```yaml
services:
  news-model:
    build: .
    command: ["./microgpt", "-load", "/data/news.json", "-mode", "serve", "-addr", ":8080"]
    volumes:
      - news-data:/data
    ports:
      - "9001:8080"

  weather-model:
    build: .
    command: ["./microgpt", "-load", "/data/weather.json", "-mode", "serve", "-addr", ":8080"]
    volumes:
      - weather-data:/data
    ports:
      - "9002:8080"

  places-model:
    build: .
    command: ["./microgpt", "-load", "/data/places.json", "-mode", "serve", "-addr", ":8080"]
    volumes:
      - places-data:/data
    ports:
      - "9003:8080"

  # Periodic trainer for the news service
  news-trainer:
    build: .
    command: >
      sh -c 'while true; do
        curl -sf https://api.example.com/news/headlines > /tmp/h.txt &&
        ./microgpt -load /data/news.json -dataset /tmp/h.txt -steps 200 -save /data/news.json;
        sleep 3600;
      done'
    volumes:
      - news-data:/data

volumes:
  news-data:
  weather-data:
  places-data:
```

### Minimal Dockerfile

```dockerfile
FROM golang:1.21-alpine AS build
WORKDIR /src
COPY go.mod main.go ./
RUN go build -o /microgpt

FROM alpine:3.19
COPY --from=build /microgpt /microgpt
ENTRYPOINT ["/microgpt"]
```

## Integration with Mu

[Mu](https://github.com/micro/mu) is a collection of focused apps — News, Chat, Places, and more — that run as a single Go binary. Each app has a built-in data source (RSS feeds, location APIs, etc.) that produces short, line-oriented text ideal for microgpt-go's character-level model.

### How It Would Work

Each Mu app can run a companion microgpt-go instance that learns the patterns of its domain data:

| Mu App    | Data Source              | microgpt-go Model Use                          |
|-----------|--------------------------|-------------------------------------------------|
| **News**  | RSS feed headlines       | Generate plausible headlines, suggest titles     |
| **Chat**  | User messages / topics   | Generate topic suggestions, auto-complete        |
| **Places**| Location names           | Generate fictional place names, fuzzy matching   |
| **Blog**  | Post titles and tags     | Suggest tags, generate title ideas               |
| **Video** | Video titles / channels  | Generate title variations, suggest search terms  |

### Data Flow

Mu apps already fetch and cache data from external APIs. That same data can be piped through microgpt-go for training:

```
  Mu App (e.g. News)
       │
       │  RSS fetch (already happens)
       ▼
  headlines.txt  ──▶  microgpt -dataset headlines.txt -steps 200 -save news.json
       │
       │  /generate API
       ▼
  "Generate a headline" ──▶ microgpt -load news.json -mode serve -addr :9001
```

Because both Mu and microgpt-go are pure Go with no external dependencies, they can share the same build toolchain, deployment pipeline, and container image.

### Connecting Mu to microgpt-go

Mu exposes an MCP (Model Context Protocol) server. A microgpt-go instance can be registered as an MCP tool so that Mu's AI features (Chat knowledge assistant, News summarizer) can call microgpt-go's `/generate` endpoint alongside larger language models:

```
Mu Chat ──▶ MCP Router ──▶ microgpt-go /generate  (domain-specific patterns)
                       ──▶ Ollama / Claude         (general reasoning)
```

This gives Mu a fast, local, zero-dependency fallback for domain-specific text generation that works even when external LLM APIs are unavailable.

## Practical Considerations

### What microgpt-go Is Good At

- **Short pattern generation** — names, titles, codes, identifiers, short phrases
- **Domain-specific vocabularies** — learning the character patterns of a specific corpus
- **Fast cold start** — loads a model and starts serving in milliseconds
- **Zero operational dependencies** — no GPU, no Python, no external services

### What It Is Not

- **Not a replacement for large language models** — it cannot reason, answer questions, or generate coherent long-form text
- **Not suitable for large vocabularies without modifications** — the character-level tokenizer works best with short, structured text

### When to Use This Pattern

Use a per-service microgpt-go model when you need:

- Synthetic data generation in the style of a service's domain (test data, placeholders, suggestions)
- A local fallback for text generation when external APIs are rate-limited or down
- Lightweight pattern learning from a continuously updating data source
- An offline-capable generation service with no external dependencies

## Summary

microgpt-go's existing features — HTTP serving, model persistence, stdin streaming, and well-known dataset support — make it a natural fit for microservice architectures where each service maintains its own small, domain-specific model. Periodic API ingestion is handled by combining standard tools (cron, curl, shell pipelines) with microgpt-go's incremental training and atomic model replacement. Integration with platforms like Mu is straightforward because both projects share the same language, deployment model, and philosophy of minimal dependencies.
