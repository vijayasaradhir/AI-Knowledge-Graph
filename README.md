# Smart Assist

Smart Assist is a lightweight knowledge graph application for extracting entities and relations from unstructured text, storing them in a graph structure, and using embeddings for similarity-based reasoning.

## Features

- Entity extraction from plain text using an OpenAI LLM with structured outputs
- Relation extraction from sentence patterns
- In-memory knowledge graph built with `networkx`
- Optional persistence to Neo4j
- TF-IDF embeddings for semantic similarity and retrieval
- Sample documents for quick testing

## Project Layout

- `smart_assist/` - application code
- `sample_data/documents.json` - tiny demo corpus
- `requirements.txt` - runtime dependencies

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` before running the extractor. You can also override the model with `KG_OPENAI_MODEL` if needed.

If the key is missing or the API call fails, the app falls back to a small heuristic extractor so the demo still works locally.

2. Run the demo:

```bash
python -m smart_assist.cli demo
```

3. Ingest custom documents from a folder or file:

```bash
python -m smart_assist.cli ingest --input sample_data
```

4. Query the graph:

```bash
python -m smart_assist.cli query --input sample_data --text "Who collaborates with OpenAI?"
```

The `query` command prints a direct answer first, then the JSON payload with the supporting graph data.
If Neo4j has already been populated, you can omit `--input` and query the persisted graph directly.

5. Run the REST service:

```bash
python -m smart_assist.server --host 127.0.0.1 --port 8000
```

The service exposes:

- `POST /ingest` with multipart file uploads or JSON `{"documents": [...]}` / `{"input": "path"}`
- `POST /query` with JSON `{"text": "..."}` and optional `top_k`, `documents`, or `input`
  - The response body is plain text containing only the answer
- `GET /health` for a simple readiness check

The UI in `smart_assist_ui` expects the backend at `http://localhost:8000` and uploads one or more files to `POST /ingest`.

## Supported Inputs

- A folder containing `.txt`, `.md`, and `.json` files
- A single `.txt`, `.md`, or `.json` file
- A JSON file can contain either one document object or a list of document objects
- JSON records may use `text`, `content`, `body`, `description`, or `summary`; if none are present, the full JSON object is used as the document text

When a folder is supplied, the loader scans it recursively and reads every supported file it finds.

## Neo4j Configuration

Set these environment variables to sync the graph to Neo4j:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (optional, defaults to `neo4j`)

If Neo4j is not configured, the app still runs fully in memory.
