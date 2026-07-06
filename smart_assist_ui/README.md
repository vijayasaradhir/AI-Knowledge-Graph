# Smart Assist Interface

React UI for the Smart Assist knowledge graph service.

## What it does

- Sends one or more files to `POST /ingest` as multipart form data.
- Sends prompt text to `POST /query` as JSON `{ "text": "..." }` and reads back a plain-text answer.
- Renders responses on the same page in a chat-style Codex-inspired layout.
- Runs all network actions asynchronously with loading states.

## Configure

Set `VITE_API_BASE_URL` if your backend is not on `http://localhost:8000`.

## Run

```bash
npm install
npm run dev
```
