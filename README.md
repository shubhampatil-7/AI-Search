# RAG Search Engine

RAG Search Engine is a Python command-line movie retrieval system built around a catalog of 5,000 Webflyx movies. It combines lexical search, dense embeddings, chunk-level semantic search, hybrid ranking, reranking, multimodal search, evaluation, and retrieval-augmented generation.

## Features

- Keyword search with stop-word removal, Porter stemming, TF-IDF, and BM25
- Semantic search using `all-MiniLM-L6-v2`
- Fixed-size word chunking and sentence-based semantic chunking
- Chunk embeddings with result aggregation by movie
- Weighted BM25/semantic hybrid search
- Reciprocal rank fusion across BM25 and semantic rankings
- OpenRouter query correction and rewriting
- Individual, batch, and cross-encoder reranking
- CLIP image search using `clip-ViT-B-32`
- Image-plus-text query rewriting through OpenRouter
- RAG answers, summaries, citations, and conversational responses
- Precision@k, recall@k, and F1 evaluation against a golden dataset
- Persistent pickle and NumPy caches

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Internet access for downloading Hugging Face models
- An `OPENROUTER_API_KEY` for OpenRouter features

## Setup

```bash
uv sync
```

For query enhancement, reranking, image query rewriting, and RAG, create `.env` in the project root:

```dotenv
OPENROUTER_API_KEY=your_api_key
```

Run all commands from the repository root.

## Project Data

| Path | Purpose |
|---|---|
| `data/movies.json` | 5,000 movie records with `id`, `title`, and `description` |
| `data/golden_dataset.json` | Evaluation queries and relevant movie titles |
| `data/stopwords.txt` | Keyword search stop words |
| `data/paddington.jpeg` | Example image for image search |
| `cache/` | Generated indexes, embeddings, document maps, and metadata |

Movie records use this shape:

```json
{
  "id": 1,
  "title": "Movie Title",
  "description": "Movie plot or synopsis"
}
```

## Quick Start

```bash
# Build and run the lexical index
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py bm25search "cute british bear marmalade"

# Search with embeddings
uv run cli/semantic_search_cli.py search "dinosaur theme park" --limit 5

# Weighted hybrid search
uv run cli/hybrid_search_cli.py weighted-search "superhero saves world" --alpha 0.5 --limit 5

# Reciprocal rank fusion
uv run cli/hybrid_search_cli.py rrf-search "zombie apocalypse" -k 60 --limit 5

# Image search
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

## CLI Reference

### Keyword Search

```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py search "romantic wedding"
uv run cli/keyword_search_cli.py tf DOC_ID TERM
uv run cli/keyword_search_cli.py idf TERM
uv run cli/keyword_search_cli.py tfidf DOC_ID TERM
uv run cli/keyword_search_cli.py bm25idf TERM
uv run cli/keyword_search_cli.py bm25tf DOC_ID TERM [K1] [B]
uv run cli/keyword_search_cli.py bm25search "talking teddy bear comedy"
```

The text pipeline converts text to lowercase, removes punctuation and stop words, then applies NLTK Porter stemming. BM25 uses `K1=1.5` and `B=0.75` by default.

### Semantic Search

```bash
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py embed_text "A bear travels to London"
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py embed_query "car racing"
uv run cli/semantic_search_cli.py search "wizards and magic" --limit 10
uv run cli/semantic_search_cli.py embed_chunks
uv run cli/semantic_search_cli.py search_chunked "children's animated bear adventure" --limit 5
```

Complete movie embeddings combine each title with its description. Chunked search splits descriptions into sentence chunks, encodes them with `all-MiniLM-L6-v2`, retains each movie's highest-scoring chunk, and ranks movies by cosine similarity.

Chunk utilities:

```bash
uv run cli/semantic_search_cli.py chunk "Text to split" --chunk-size 200 --overlap 1
uv run cli/semantic_search_cli.py semantic_chunk "First sentence. Second sentence." --max-chunk-size 4 --overlap 0
```

### Hybrid Search

Weighted search normalizes BM25 and semantic scores before combining them:

```bash
uv run cli/hybrid_search_cli.py weighted-search "car racing" --alpha 0.5 --limit 5
```

```text
hybrid = alpha * BM25 + (1 - alpha) * semantic
```

- `--alpha 1.0`: BM25 only
- `--alpha 0.5`: balanced
- `--alpha 0.0`: semantic only

Reciprocal rank fusion combines rankings without comparing their raw scores:

```bash
uv run cli/hybrid_search_cli.py rrf-search "zombie apocalypse" -k 60 --limit 5
```

```text
RRF = sum(1 / (k + rank))
```

Optional query enhancement and reranking:

```bash
--enhance spell
--enhance rewrite
--rerank-method individual
--rerank-method batch
--rerank-method cross_encoder
--evaluate
```

Example:

```bash
uv run cli/hybrid_search_cli.py rrf-search "superhero saves world" \
  --enhance rewrite \
  --rerank-method cross_encoder \
  --evaluate
```

- `individual` scores documents individually with the LLM from 0–10.
- `batch` asks the LLM for a JSON-ranked movie ID list.
- `cross_encoder` uses `cross-encoder/ms-marco-TinyBERT-L2-v2` locally.
- `--evaluate` asks an LLM to score displayed results from 0–3.

Score lists can also be normalized:

```bash
uv run cli/hybrid_search_cli.py normalize 2 4 6 10
```

### Multimodal Image Search

```bash
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

`verify_image_embedding` reports the CLIP embedding size. `image_search` encodes movie titles and descriptions with `clip-ViT-B-32`, compares them with the image embedding using cosine similarity, and returns ranked movie results.

### Image-to-Text Query Rewriting

```bash
uv run cli/describe_image_cli.py \
  --image data/paddington.jpeg \
  --query "find a matching family movie"
```

This utility sends the image and query to an OpenRouter vision model using `openrouter/free`, then prints a rewritten movie-search query.

### Retrieval-Augmented Generation

```bash
uv run cli/augmented_generation_cli.py rag "What movies feature a magical bear?"
uv run cli/augmented_generation_cli.py summarize "Family movies with animals" --limit 8
uv run cli/augmented_generation_cli.py citations "Which movies are about dinosaurs?" --limit 5
uv run cli/augmented_generation_cli.py question "What is good for family night?" --limit 5
```

| Command | Purpose |
|---|---|
| `rag` | Retrieve five movies and generate an answer |
| `summarize` | Synthesize retrieved movies into a concise summary |
| `citations` | Generate a numbered, cited answer |
| `question` | Produce a conversational answer |

Each command retrieves supporting movies with RRF and sends the query plus retrieved documents to OpenRouter using `openrouter/free`.

### Search Evaluation

```bash
uv run cli/evaluation_cli.py --limit 5
uv run cli/evaluation_cli.py --limit 10
```

Evaluation uses `data/golden_dataset.json` and reports precision@k, recall@k, F1, retrieved titles, and relevant titles for every test query:

```text
Precision@k = relevant results in top k / k
Recall@k    = relevant results in top k / total relevant
F1          = 2 × Precision × Recall / (Precision + Recall)
```

## Generated Caches

| File | Content |
|---|---|
| `cache/index.pkl` | Inverted keyword index |
| `cache/docmap.pkl` | Movie ID-to-movie map |
| `cache/doc_lengths.pkl` | Document lengths |
| `cache/term_frequencies.pkl` | Term-frequency counters |
| `cache/movie_embeddings.npy` | Complete movie embeddings |
| `cache/chunk_embeddings.npy` | Movie chunk embeddings |
| `cache/chunk_metadata.json` | Movie and chunk metadata |

Remove caches to force regeneration:

```bash
rm -f cache/*.pkl cache/*.npy cache/chunk_metadata.json
```

Hybrid search can rebuild the keyword index automatically. Semantic search rebuilds missing or mismatched movie embeddings.

## Project Structure

```text
cli/
├── keyword_search.py / keyword_search_cli.py
├── semantic_search_cli.py
├── hybrid_search.py / hybrid_search_cli.py
├── multimodal_search_cli.py
├── augmented_generation_cli.py
├── evaluation_cli.py
├── describe_image_cli.py
├── search_utils.py
└── lib/
    ├── semantic_search.py
    ├── multimodal_search.py
    ├── query_enhancement.py
    ├── reranking.py
    ├── augmentation.py
    └── evaluation.py
```

## Troubleshooting

**Missing OpenRouter key:** add `OPENROUTER_API_KEY` to `.env`.

**Missing keyword index:**

```bash
uv run cli/keyword_search_cli.py build
```

**Missing or stale embeddings:**

```bash
rm -f cache/movie_embeddings.npy cache/chunk_embeddings.npy cache/chunk_metadata.json
```

The first semantic or cross-encoder run may download model files and take longer. Run scripts from the project root with `uv run`, and keep `.env` plus generated cache files private.
