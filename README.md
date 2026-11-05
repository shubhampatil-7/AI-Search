# RAG Search Engine

A Python command-line movie search and retrieval system for the movie catalog. It combines keyword ranking, dense semantic search, reciprocal rank fusion, query rewriting, reranking, multimodal image search, retrieval evaluation, and Retrieval-Augmented Generation.

The project includes a 5,000-movie catalog in `data/movies.json`, keyword and embedding caches in `cache/`, an evaluation dataset in `data/golden_dataset.json`, and several focused CLI tools under `cli/`.

## Table of Contents

- [Features](#features)
- [Search Methods](#search-methods)
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [Keyword Search](#keyword-search)
  - [Semantic Search](#semantic-search)
  - [Hybrid Search](#hybrid-search)
  - [Multimodal Image Search](#multimodal-image-search)
  - [Image-to-Text Query Rewriting](#image-to-text-query-rewriting)
  - [Retrieval-Augmented Generation](#retrieval-augmented-generation)
  - [Search Evaluation](#search-evaluation)
- [Data and Caching](#data-and-caching)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

## Features

- Keyword search with tokenization, stop-word removal, stemming, TF-IDF, and BM25
- Dense semantic search with cosine similarity
- Fixed-size and sentence-based document chunking
- Chunk-level embeddings and movie-level result aggregation
- Weighted hybrid search combining normalized BM25 and semantic scores
- Reciprocal rank fusion (RRF) over BM25 and semantic rankings
- LLM-based and local cross-encoder reranking
- LLM query spelling correction, rewriting, and expansion
- CLIP image embeddings and image-to-movie ranking
- Image plus text query rewriting through OpenRouter
- Retrieval-augmented answers, summaries, citations, and conversational responses
- Precision@k, recall@k, and F1 evaluation against a golden dataset
- Persistent `pickle` and NumPy caches

## Search Methods

| Method | Model or algorithm | Use case |
|---|---|---|
| Keyword / BM25 | Porter stemming + BM25 | Exact and lexical terms |
| Semantic | `all-MiniLM-L6-v2` | Meaning-based text retrieval |
| Chunked semantic | `all-MiniLM-L6-v2` + sentence chunks | Long movie descriptions |
| Weighted hybrid | Min-max normalized BM25 and semantic scores | Tunable blend of lexical and semantic relevance |
| RRF hybrid | Reciprocal rank fusion | Rank aggregation without score normalization |
| Image search | `clip-ViT-B-32` | Find movies using a photographic image |
| Reranking | LLM, or `cross-encoder/ms-marco-TinyBERT-L2-v2` | Improve the top results |
| RAG | Hybrid retrieval + OpenRouter model | Generate answers grounded in retrieved movies |

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Internet access when downloading Sentence Transformers models or Hugging Face models for the first time
- An OpenRouter API key for LLM query enhancement, reranking, evaluation, image query rewriting, and RAG commands

Core dependencies are declared in `pyproject.toml`:

- NumPy
- NLTK
- Pillow
- Sentence Transformers
- Transformers
- OpenAI Python client
- python-dotenv

## Installation

Run all commands from the repository root.

```bash
git clone <repository-url>
cd rag-search-engine
uv sync
```

The project uses Python 3.12+ as defined by `pyproject.toml`.

NLTK stemming does not require a separate NLTK data download for the current implementation; `PorterStemmer` is used directly. The first use of a Sentence Transformers or cross-encoder model may download model files from Hugging Face.

## Environment Variables

Create a `.env` file in the repository root:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
```

The application currently sends OpenRouter requests to:

```text
https://openrouter.ai/api/v1
```

It uses `openrouter/free` as the default model for query enhancement, reranking, LLM evaluation, image query rewriting, and RAG generation.

Commands that use these modules require the key to be present. The keyword, semantic, and local image-search paths generally do not need it.

## Quick Start

Build the inverted index:

```bash
uv run cli/keyword_search_cli.py build
```

Run BM25 search:

```bash
uv run cli/keyword_search_cli.py bm25search "cute british bear marmalade"
```

Run document semantic search:

```bash
uv run cli/semantic_search_cli.py search "dinosaur theme park" --limit 5
```

Run weighted hybrid search:

```bash
uv run cli/hybrid_search_cli.py weighted-search "superhero saves the world" --alpha 0.5 --limit 5
```

Run hybrid search with rank fusion:

```bash
uv run cli/hybrid_search_cli.py rrf-search "zombie apocalypse" -k 60 --limit 5
```

## CLI Reference

### Keyword Search

```bash
uv run cli/keyword_search_cli.py <command> [arguments]
```

Available commands:

| Command | Arguments | Description |
|---|---|---|
| `build` | None | Builds the inverted index and saves keyword statistics to `cache/` |
| `search` | `query` | Simple OR-style stemming and stop-word search |
| `tf` | `doc_id term` | Raw term frequency for one document |
| `idf` | `term` | Logarithmic smoothed IDF |
| `tfidf` | `doc_id term` | Raw TF multiplied by IDF |
| `bm25idf` | `term` | BM25 IDF for a term |
| `bm25tf` | `doc_id term [k1] [b]` | BM25 term score with optional K1 and B overrides |
| `bm25search` | `query` | Full BM25 search; defaults to five results |

Examples:

```bash
uv run cli/keyword_search_cli.py search "romantic wedding"
uv run cli/keyword_search_cli.py tf 76 paddington
uv run cli/keyword_search_cli.py idf bear
uv run cli/keyword_search_cli.py tfidf 76 bear
uv run cli/keyword_search_cli.py bm25idf bear
uv run cli/keyword_search_cli.py bm25tf 76 bear
uv run cli/keyword_search_cli.py bm25tf 76 bear 1.5 0.75
uv run cli/keyword_search_cli.py bm25search "talking teddy bear comedy"
```

`search`, `tf`, `idf`, `tfidf`, `bm25idf`, and `bm25tf` load the cached index, so run `build` first. The index combines each movie title and description before tokenization.

Text preprocessing:

1. Converts text to lowercase.
2. Removes punctuation.
3. Removes words from `data/stopwords.txt`.
4. Applies NLTK `PorterStemmer`.

BM25 defaults are:

```text
K1 = 1.5
B  = 0.75
```

### Semantic Search

```bash
uv run cli/semantic_search_cli.py <command> [arguments]
```

Available commands:

| Command | Arguments | Description |
|---|---|---|
| `verify` | None | Loads and reports the `all-MiniLM-L6-v2` model |
| `embed_text` | `text` | Generates one embedding and displays its first dimensions |
| `verify_embeddings` | None | Builds or loads movie embeddings and reports the matrix shape |
| `embed_query` | `query` | Generates one query embedding |
| `search` | `query --limit N` | Searches complete movie title and description embeddings |
| `chunk` | `text --chunk-size N --overlap N` | Splits text into fixed-size word chunks |
| `semantic_chunk` | `text --max-chunk-size N --overlap N` | Splits text at sentence boundaries |
| `embed_chunks` | None | Builds or loads chunk embeddings for movie descriptions |
| `search_chunked` | `query --limit N` | Searches chunk embeddings and returns the best matching movie |

Examples:

```bash
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py embed_text "A brave bear travels to London"
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py embed_query "car racing"
uv run cli/semantic_search_cli.py search "wizards and magic" --limit 10
uv run cli/semantic_search_cli.py chunk "One two three four five." --chunk-size 2 --overlap 1
uv run cli/semantic_search_cli.py semantic_chunk "First sentence. Second sentence. Third sentence." --max-chunk-size 2 --overlap 0
uv run cli/semantic_search_cli.py embed_chunks
uv run cli/semantic_search_cli.py search_chunked "children's animated bear adventure" --limit 5
```

Document embeddings are generated from:

```text
<movie title>: <movie description>
```

Chunked embeddings use descriptions only and split them into groups of up to four sentences, with overlap controlled by the chunk configuration. The chunk searcher records each movie's highest-scoring chunk and returns movie-level results.

Default chunk settings:

```text
Fixed chunk size: 200 words
Fixed overlap: 1 word
Semantic max chunk size: 4 sentences
Standalone `semantic_chunk` overlap: 0 by default
Chunked-search embedding overlap: 1
```

### Hybrid Search

```bash
uv run cli/hybrid_search_cli.py <command> [arguments]
```

#### Normalize scores

Normalize a list using min-max scaling:

```bash
uv run cli/hybrid_search_cli.py normalize 2 4 6 10
```

If every input has the same value, each normalized score is returned as `1.0`.

#### Weighted hybrid search

```bash
uv run cli/hybrid_search_cli.py weighted-search "query" --alpha 0.5 --limit 5
```

`--alpha` controls the blend:

| Alpha | Result |
|---|---|
| `1.0` | BM25 only |
| `0.5` | Equal weighted blend |
| `0.0` | Semantic only |

BM25 and semantic scores are normalized independently before combination. The combined score is:

```text
hybrid = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

The search expands each underlying ranking to 500 times the requested limit before fusing movie-level matches.

#### Reciprocal rank fusion

```bash
uv run cli/hybrid_search_cli.py rrf-search "query" -k 60 --limit 5
```

RRF combines ranks from the BM25 and chunked semantic results:

```text
RRF score = sum(1 / (k + rank))
```

Available query enhancement options:

```bash
--enhance spell
--enhance rewrite
```

Query enhancement uses OpenRouter and requires `OPENROUTER_API_KEY`.

Available reranking options:

```bash
--rerank-method individual
--rerank-method batch
--rerank-method cross_encoder
```

Reranking behavior:

- `individual`: asks the LLM to score each result from 0–10
- `batch`: asks the LLM to return a JSON array of movie IDs in relevance order
- `cross_encoder`: ranks query/document pairs locally with `cross-encoder/ms-marco-TinyBERT-L2-v2`

Evaluate the reranked results with the LLM:

```bash
uv run cli/hybrid_search_cli.py rrf-search "superhero saves the world" \
  --rerank-method cross_encoder \
  --evaluate
```

The `--evaluate` flag asks an OpenRouter model to assign every displayed result a relevance score from 0–3.

### Multimodal Image Search

```bash
uv run cli/multimodal_search_cli.py <command> [arguments]
```

Available commands:

| Command | Arguments | Description |
|---|---|---|
| `verify_image_embedding` | `image` | Encodes the image and prints the embedding dimensions |
| `image_search` | `image` | Ranks movies by cosine similarity between image and movie text embeddings |

Examples:

```bash
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

The multimodal implementation uses:

- `clip-ViT-B-32` from Sentence Transformers
- Movie text encoded as `<title>: <description>`
- Cosine similarity between the image embedding and each movie text embedding

Image search computes its text embeddings on every run. No separate image embedding cache is currently generated.

The default image search path is `data/paddington.jpeg` if the function is invoked directly without an argument.

### Image-to-Text Query Rewriting

The project also includes `cli/describe_image_cli.py`, a utility that combines an image and a text query and asks an OpenRouter vision-capable model to rewrite the query for movie search.

It uses:

```text
Model: openrouter/free
```

It reads the image as a base64 data URL and sends both the image and instructions to OpenRouter. It requires `OPENROUTER_API_KEY`.

Run from the repository root with:

```bash
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "find a matching family movie"
```

The model is instructed to synthesize the visual and textual information, focus on movie-specific details, and return only the rewritten query.

### Retrieval-Augmented Generation

```bash
uv run cli/augmented_generation_cli.py <command> [arguments]
```

Available commands:

| Command | Arguments | Description |
|---|---|---|
| `rag` | `query` | Retrieves five movies with RRF and generates an answer |
| `summarize` | `query --limit N` | Synthesizes retrieved movie information into a 3–4 sentence summary |
| `citations` | `query --limit N` | Generates a sourced answer using numbered references |
| `question` | `question --limit N` | Produces a casual conversational answer from retrieved context |

Examples:

```bash
uv run cli/augmented_generation_cli.py rag "what movies feature a magical bear?"
uv run cli/augmented_generation_cli.py summarize "family movies with animals" --limit 8
uv run cli/augmented_generation_cli.py citations "which movies are about dinosaurs?" --limit 5
uv run cli/augmented_generation_cli.py question "What is the best movie for a family night?" --limit 5
```

These commands retrieve context through reciprocal rank fusion before creating prompts. Generation uses `openrouter/free` and OpenRouter API credentials from `.env`.

The `rag` command currently fixes its retrieval limit to five. Other RAG subcommands expose a configurable `--limit`.

### Search Evaluation

```bash
uv run cli/evaluation_cli.py --limit 5
```

The evaluation command:

1. Loads 10 queries from `data/golden_dataset.json`.
2. Loads or creates the movie, chunk, and keyword caches.
3. Runs RRF hybrid search for each query.
4. Compares retrieved titles with the relevant titles in the golden dataset.
5. Prints precision@k, recall@k, and F1.

Example:

```bash
uv run cli/evaluation_cli.py --limit 10
```

`precision@k`, `recall@k`, and `f1_score` use the following formulas:

```text
precision@k = relevant items in top k / k
recall@k   = relevant items in top k / relevant items
F1         = 2 * precision * recall / (precision + recall)
```

The CLI also prints retrieved and relevant movie titles for every query.

## Data and Caching

### Source data

```text
data/movies.json           5,000 movie titles and descriptions
data/golden_dataset.json   10 retrieval evaluation test cases
data/stopwords.txt         198 keyword stop words
data/paddington.jpeg       sample image for multimodal search
```

Movie records have the shape:

```json
{
  "id": 1,
  "title": "Movie Title",
  "description": "Movie plot or synopsis"
}
```

Golden records have the shape:

```json
{
  "query": "search text",
  "relevant_docs": ["Movie Title"]
}
```

### Generated cache files

| File | Purpose |
|---|---|
| `cache/index.pkl` | Inverted term-to-document index |
| `cache/docmap.pkl` | Movie ID-to-movie mapping |
| `cache/doc_lengths.pkl` | Number of tokens in each movie document |
| `cache/term_frequencies.pkl` | Per-document term-frequency counters |
| `cache/movie_embeddings.npy` | Dense embeddings for complete movies |
| `cache/chunk_embeddings.npy` | Dense embeddings for movie description chunks |
| `cache/chunk_metadata.json` | Chunk index, movie index, and total chunks |

The cache directory is generated automatically when needed. Delete caches to force regeneration:

```bash
rm -f cache/*.pkl cache/*.npy cache/chunk_metadata.json
```

Cache behavior:

- `build` always rebuilds the inverted index.
- Movie embeddings are reused when the cache length matches the current catalog.
- Chunk embeddings are reused when both chunk cache files exist.
- The keyword index is rebuilt automatically by hybrid search if `cache/index.pkl` is missing.

## Project Structure

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   ├── movies.json
│   ├── golden_dataset.json
│   ├── stopwords.txt
│   └── paddington.jpeg
├── cache/
│   ├── index.pkl
│   ├── docmap.pkl
│   ├── doc_lengths.pkl
│   ├── term_frequencies.pkl
│   ├── movie_embeddings.npy
│   ├── chunk_embeddings.npy
│   └── chunk_metadata.json
└── cli/
    ├── keyword_search.py
    ├── keyword_search_cli.py
    ├── hybrid_search.py
    ├── hybrid_search_cli.py
    ├── semantic_search_cli.py
    ├── multimodal_search_cli.py
    ├── augmented_generation_cli.py
    ├── evaluation_cli.py
    ├── describe_image_cli.py
    ├── search_utils.py
    ├── test_llm.py
    └── lib/
        ├── semantic_search.py
        ├── multimodal_search.py
        ├── query_enhancement.py
        ├── reranking.py
        ├── evaluation.py
        └── augmentation.py
```

## How It Works

### Inverted index

`InvertedIndex.build()` processes each movie title and description, creates an index of stemmed terms to document IDs, and records document lengths and term frequencies. The resulting state is serialized to the four keyword cache files.

### BM25

BM25 combines:

- Term frequency adjusted by document length.
- BM25 IDF computed from document frequency.
- K1, which controls term-frequency saturation.
- B, which controls document-length normalization.

### Semantic embeddings

The semantic subsystem encodes title-plus-description text with `all-MiniLM-L6-v2`. Cosine similarity ranks all movies for a query. Chunked semantic search instead embeds description chunks, scores every chunk, and retains each movie's best chunk score.

### Hybrid ranking

Weighted search normalizes each ranking independently and calculates one combined score per movie. RRF assigns scores from rank positions instead of raw similarity values, making it suitable when BM25 and semantic rankings use different scales.

### Multimodal ranking

CLIP encodes an input image and all movie title-description strings into a shared embedding space. Cosine similarity ranks the movie texts against the image.

### Retrieval-augmented generation

The RAG commands first use RRF retrieval to find supporting movie documents. They insert those documents and the user's request into a prompt sent through OpenRouter.

## Troubleshooting

### `OPENROUTER_API_KEY` is not set

Create `.env` in the repository root:

```dotenv
OPENROUTER_API_KEY=your_key
```

Then rerun the command. Keep `.env` private and never commit API credentials.

### Missing keyword cache

Build the index:

```bash
uv run cli/keyword_search_cli.py build
```

### Missing or stale embeddings

Remove only the generated cache files you want to rebuild, then rerun the relevant command:

```bash
rm -f cache/movie_embeddings.npy
rm -f cache/chunk_embeddings.npy cache/chunk_metadata.json
```

Embeddings can also be rebuilt explicitly:

```bash
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py embed_chunks
```

### Slow first run

Sentence Transformers and cross-encoder models are downloaded when first loaded. Subsequent runs reuse local Hugging Face model caches. Image search also encodes every movie description on each invocation unless run with a persistent application process.

### Module import errors

Run scripts from the repository root and use `uv run`:

```bash
cd rag-search-engine
uv run cli/keyword_search_cli.py --help
```

Do not invoke modules as arbitrary package paths without the repository root on the Python path.

### LLM response parsing errors

Batch reranking expects a raw JSON array of integer movie IDs. Individual reranking expects exactly one integer from 0–10. If an OpenRouter model adds prose or invalid JSON, retry the command or use the cross-encoder reranker.

### No search results

- Check that `data/movies.json` contains the expected records.
- Build the keyword index.
- Try broader terms, weighted search, semantic search, or RRF.
- Delete stale embedding caches if the catalog has changed.

## Development Notes

- Cache files and local `.env` files should not be committed.
- The catalog and golden dataset are project data.
- Model downloads require initial network access.
- Hybrid search loads chunk embeddings and may require substantial memory for the 5,000-item catalog.
- BM25 term commands require a single-token term; multiword terms raise a tokenization error.
- Search preview text is typically limited to 100 characters in CLI output, while the full descriptions remain available to ranking and prompts.
