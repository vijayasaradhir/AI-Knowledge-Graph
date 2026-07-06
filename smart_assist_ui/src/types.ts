export type EndpointResult = {
  ok: boolean;
  status: number;
  contentType: string;
  payload: unknown;
  rawText?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  details?: string;
};

export type IngestSummary = {
  uploadedFiles: string[];
  result: EndpointResult | null;
};
