export type ProductSummary = {
  id: string;
  product_id: string;
  external_id: string;
  title: string;
  source_type: string;
  source_url: string | null;
  metadata: Record<string, unknown>;
  counts: {
    sections?: number;
    reviews?: number;
    chunks?: number;
  };
  created_at: string | null;
  updated_at: string | null;
};

export type ProductSection = {
  id: string;
  source_id: string;
  section_type: string;
  heading: string | null;
  text: string;
  ordinal: number;
  metadata: Record<string, unknown>;
};

export type Review = {
  id: string;
  source_id: string;
  title: string | null;
  body: string;
  rating: number | null;
  date: string | null;
  verified: boolean;
  helpful_votes: number;
  metadata: Record<string, unknown>;
};

export type Chunk = {
  id: string;
  source_id: string;
  source_type: string;
  text: string;
  token_count: number | null;
  metadata: Record<string, unknown>;
};

export type ProductDetail = ProductSummary & {
  sections: ProductSection[];
  reviews: Review[];
  chunks: Chunk[];
};

export type IngestionJob = {
  job_id: string;
  product_id: string | null;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  source_type: string;
  input_uri: string | null;
  error_message: string | null;
  stats: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
};

export type Evidence = {
  chunk_id?: string;
  score: number;
  text: string;
  source_id: string;
  source_type: string;
  meta: Record<string, unknown>;
  score_components?: Record<string, number>;
};

export type AskResponse = {
  question: string;
  query_log_id?: string;
  result: {
    answer: string;
    engine: string;
    confidence: number;
    citations?: Array<{
      source_id: string;
      source_type: string;
      meta: Record<string, unknown>;
    }>;
  };
  evidence: Evidence[];
  consensus: {
    summary?: string;
    by_window?: Array<{
      window: string;
      n: number;
      pos: number;
      neg: number;
      neu: number;
      pos_rate: number;
      neg_rate: number;
    }>;
    overall?: Record<string, number>;
  };
  faithfulness: {
    overall_supported_rate?: number;
    sentences?: Array<{
      sentence: string;
      supported: boolean;
      cited_ids: string[];
      notes: string;
    }>;
  };
  diagnostics: {
    latency_ms?: number;
    retrieval?: Record<string, unknown>;
    answer?: Record<string, unknown>;
  };
};

export type EvalDataset = {
  id: string;
  name: string;
  description: string | null;
  metadata: Record<string, unknown>;
  example_count: number | null;
  created_at: string | null;
  examples?: Array<{
    id: string;
    product_id: string;
    question: string;
    answerable: boolean;
    gold_spans: Array<Record<string, unknown>>;
  }>;
};

export type EvalRun = {
  id: string;
  dataset_id: string;
  model: string;
  status: "running" | "succeeded" | "failed";
  metrics: Record<string, unknown>;
  error_message: string | null;
  created_at: string | null;
  completed_at: string | null;
};

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

const BACKEND_PREFIX = "/api/backend";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BACKEND_PREFIX}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "error" in payload
        ? String((payload as { error?: { message?: string } }).error?.message)
        : typeof payload === "string"
          ? payload
          : `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}

function postJson<T>(path: string, body: Record<string, JsonValue>) {
  return requestJson<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export const api = {
  health: () => requestJson<{ status: string; environment?: string }>("/healthz"),
  products: () => requestJson<{ products: ProductSummary[] }>("/api/v1/products"),
  product: (productId: string) => requestJson<ProductDetail>(`/api/v1/products/${productId}`),
  jobs: () => requestJson<{ jobs: IngestionJob[] }>("/api/v1/ingestion-jobs"),
  createJob: (body: {
    product_html: string;
    reviews_file: string;
    source_url?: string;
    enqueue: boolean;
  }) => postJson<IngestionJob>("/api/v1/ingestion-jobs", body),
  ask: (body: { product_id: string; question: string; model?: string }) =>
    postJson<AskResponse>("/api/v1/ask", body),
  datasets: () => requestJson<{ datasets: EvalDataset[] }>("/api/v1/eval-datasets"),
  createDataset: (body: {
    name: string;
    description?: string;
    source_path?: string;
    replace?: boolean;
  }) => postJson<EvalDataset>("/api/v1/eval-datasets", body),
  runs: () => requestJson<{ runs: EvalRun[] }>("/api/v1/eval-runs"),
  createRun: (body: { dataset_name?: string; dataset_id?: string; model: string; k: number }) =>
    postJson<EvalRun>("/api/v1/eval-runs", body),
};
