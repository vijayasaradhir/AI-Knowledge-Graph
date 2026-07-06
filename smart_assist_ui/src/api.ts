import type { EndpointResult } from "./types";

const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.replace(/\/+$/, "");
}

async function readResponse(response: Response): Promise<EndpointResult> {
  const contentType = response.headers.get("content-type") || "";
  const rawText = await response.text();
  let payload: unknown = rawText;

  if (contentType.includes("application/json")) {
    try {
      payload = JSON.parse(rawText);
    } catch {
      payload = rawText;
    }
  }

  return {
    ok: response.ok,
    status: response.status,
    contentType,
    payload,
    rawText,
  };
}

export async function ingestFiles(files: File[], baseUrl: string = DEFAULT_BASE_URL): Promise<EndpointResult> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file, file.name);
  }

  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/ingest`, {
    method: "POST",
    body: formData,
  });

  return readResponse(response);
}

export async function queryAssist(text: string, baseUrl: string = DEFAULT_BASE_URL): Promise<EndpointResult> {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  return readResponse(response);
}
