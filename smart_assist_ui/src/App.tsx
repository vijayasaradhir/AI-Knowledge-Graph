import { type ChangeEvent, type FormEvent, useMemo, useRef, useState } from "react";
import { ingestFiles, queryAssist } from "./api";
import type { ChatMessage, EndpointResult, IngestSummary } from "./types";

const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function describeResult(result: EndpointResult | null) {
  if (!result) {
    return "No response yet.";
  }
  if (typeof result.payload === "string") {
    return result.payload;
  }
  return JSON.stringify(result.payload, null, 2);
}

function formatStatus(result: EndpointResult | null) {
  if (!result) return "Idle";
  return `${result.ok ? "Success" : "Error"} ${result.status}`;
}

function newId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

export default function App() {
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL);
  const [queryText, setQueryText] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Upload documents to build the graph, then ask a question using the prompt bar below. I will keep the conversation grounded in the endpoint response.",
    },
  ]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [ingestSummary, setIngestSummary] = useState<IngestSummary>({ uploadedFiles: [], result: null });
  const [ingestLoading, setIngestLoading] = useState(false);
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const lastAssistantMessage = [...messages].reverse().find((message) => message.role === "assistant");
  const hasFiles = selectedFiles.length > 0;

  const stats = useMemo(
    () => [
      { label: "Queued files", value: String(selectedFiles.length) },
      { label: "Ingest status", value: formatStatus(ingestSummary.result) },
      { label: "Last reply", value: lastAssistantMessage ? "Ready" : "Waiting" },
    ],
    [ingestSummary.result, lastAssistantMessage, selectedFiles.length],
  );

  async function handleIngest(event: FormEvent) {
    event.preventDefault();
    if (!selectedFiles.length || ingestLoading) return;

    setError(null);
    setIngestLoading(true);
    try {
      const result = await ingestFiles(selectedFiles, baseUrl);
      setIngestSummary({
        uploadedFiles: selectedFiles.map((file) => file.name),
        result,
      });
      setMessages((current) => [
        ...current,
        {
          id: newId("ingest"),
          role: "assistant",
          content: result.ok
            ? `Ingest completed successfully for ${selectedFiles.length} file(s).`
            : `Ingest failed with status ${result.status}.`,
          details: describeResult(result),
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Ingest request failed.";
      setError(message);
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleQuery(event: FormEvent) {
    event.preventDefault();
    const text = queryText.trim();
    if (!text || queryLoading) return;

    setError(null);
    setQueryLoading(true);
    setMessages((current) => [
      ...current,
      {
        id: newId("user"),
        role: "user",
        content: text,
      },
    ]);
    setQueryText("");

    try {
      const result = await queryAssist(text, baseUrl);
      setMessages((current) => [
        ...current,
        {
          id: newId("assistant"),
          role: "assistant",
          content: result.ok ? describeResult(result) : `Query failed with status ${result.status}.`,
          details: result.ok ? undefined : describeResult(result),
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Query request failed.";
      setError(message);
    } finally {
      setQueryLoading(false);
    }
  }

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []);
    setSelectedFiles(files);
  }

  function clearFiles() {
    setSelectedFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  return (
    <div className="shell">
      <div className="aurora aurora-left" />
      <div className="aurora aurora-right" />

      <header className="topbar">
        <div>
          <p className="eyebrow">Smart Assist</p>
          <h1>Smart Assist Interface</h1>
          <p className="lede">
            A Codex-inspired workspace for async document ingest and graph queries.
          </p>
        </div>
        <div className="endpoint-card">
          <label htmlFor="baseUrl">API base URL</label>
          <input
            id="baseUrl"
            value={baseUrl}
            onChange={(event) => setBaseUrl(event.target.value)}
            placeholder="http://localhost:8000"
            spellCheck={false}
          />
        </div>
      </header>

      <main className="layout">
        <section className="sidebar">
          <div className="panel">
            <div className="panel-head">
              <h2>Ingest</h2>
              <span className={ingestLoading ? "badge badge-warm" : "badge"}>{ingestLoading ? "Uploading" : "Idle"}</span>
            </div>
            <form onSubmit={handleIngest} className="stack">
              <label className="file-dropzone">
                <input ref={fileInputRef} type="file" multiple onChange={handleFileSelection} />
                <span className="drop-title">Drop files here or browse</span>
                <span className="drop-subtitle">Supports one or more files sent as multipart form data.</span>
              </label>
              <div className="file-actions">
                <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()}>
                  Browse files
                </button>
                <button type="button" className="ghost-button" onClick={clearFiles} disabled={!hasFiles}>
                  Clear
                </button>
              </div>
              <button type="submit" className="primary-button" disabled={!hasFiles || ingestLoading}>
                {ingestLoading ? "Uploading..." : "Run ingest"}
              </button>
            </form>

            <div className="file-list">
              <h3>Selected files</h3>
              {hasFiles ? (
                <ul>
                  {selectedFiles.map((file) => (
                    <li key={`${file.name}-${file.size}`}>
                      <span>{file.name}</span>
                      <span>{Math.ceil(file.size / 1024)} KB</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No files selected yet.</p>
              )}
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h2>Session</h2>
              <span className="badge">Async</span>
            </div>
            <div className="stats">
              {stats.map((item) => (
                <div key={item.label} className="stat-card">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="chat">
            <div className="chat-head">
              <div>
                <p className="eyebrow">Query</p>
                <h2>Prompt composer</h2>
              </div>
              <span className={queryLoading ? "badge badge-cool" : "badge"}>{queryLoading ? "Thinking" : "Ready"}</span>
            </div>

            <div className="message-stream" aria-live="polite">
              {messages.map((message) => (
                <article key={message.id} className={`message message-${message.role}`}>
                  <div className="message-meta">{message.role === "user" ? "You" : "Smart Assist"}</div>
                  <p>{message.content}</p>
                  {message.details ? <pre>{message.details}</pre> : null}
                </article>
              ))}
            </div>

            <form className="composer" onSubmit={handleQuery}>
              <textarea
                value={queryText}
                onChange={(event) => setQueryText(event.target.value)}
                placeholder="Ask something about the ingested graph..."
                rows={3}
              />
              <div className="composer-actions">
                <p>Press Enter or click Send to query the `/query` endpoint.</p>
                <button type="submit" className="primary-button" disabled={!queryText.trim() || queryLoading}>
                  {queryLoading ? "Sending..." : "Send"}
                </button>
              </div>
            </form>
          </div>
        </section>
      </main>

      {error ? <div className="toast">{error}</div> : null}
    </div>
  );
}
