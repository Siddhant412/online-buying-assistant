"use client";

import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  FileText,
  FlaskConical,
  Loader2,
  MessageSquare,
  PackageSearch,
  Play,
  RefreshCw,
  Search,
  Send,
  UploadCloud,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import {
  api,
  AskResponse,
  EvalDataset,
  EvalRun,
  IngestionJob,
  ProductDetail,
  ProductSummary,
} from "@/lib/api";

type View = "library" | "ingestion" | "ask" | "evals";

type LoadState = {
  loading: boolean;
  message: string | null;
  error: string | null;
};

const emptyLoadState: LoadState = { loading: false, message: null, error: null };

const views: Array<{ id: View; label: string; icon: React.ElementType }> = [
  { id: "library", label: "Library", icon: PackageSearch },
  { id: "ingestion", label: "Ingestion", icon: UploadCloud },
  { id: "ask", label: "Ask", icon: MessageSquare },
  { id: "evals", label: "Evaluations", icon: FlaskConical },
];

function formatDate(value: string | null | undefined) {
  if (!value) return "n/a";
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function compactJson(value: unknown) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value, null, 2);
}

function statusClass(status: string) {
  if (["succeeded", "ok"].includes(status)) return "good";
  if (["failed", "error"].includes(status)) return "bad";
  if (["running", "queued"].includes(status)) return "warn";
  return "neutral";
}

function StatusPill({ status }: { status: string }) {
  return <span className={`status ${statusClass(status)}`}>{status}</span>;
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty">
      <FileText size={28} />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function Notice({ state }: { state: LoadState }) {
  if (!state.error && !state.message && !state.loading) return null;
  return (
    <div className={`notice ${state.error ? "bad" : state.loading ? "warn" : "good"}`}>
      {state.loading ? <Loader2 className="spin" size={16} /> : state.error ? <AlertCircle size={16} /> : <CheckCircle2 size={16} />}
      <span>{state.error ?? state.message ?? "Loading"}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value ?? "n/a"}</strong>
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<View>("library");
  const [loadState, setLoadState] = useState<LoadState>(emptyLoadState);
  const [apiStatus, setApiStatus] = useState("unknown");
  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [productDetail, setProductDetail] = useState<ProductDetail | null>(null);
  const [selectedProductId, setSelectedProductId] = useState("");
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [datasets, setDatasets] = useState<EvalDataset[]>([]);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [answer, setAnswer] = useState<AskResponse | null>(null);

  const selectedProduct = useMemo(
    () => products.find((product) => product.product_id === selectedProductId) ?? products[0],
    [products, selectedProductId],
  );

  const refreshAll = useCallback(async () => {
    setLoadState({ loading: true, message: null, error: null });
    try {
      const [health, productPayload, jobPayload, datasetPayload, runPayload] = await Promise.all([
        api.health(),
        api.products(),
        api.jobs(),
        api.datasets(),
        api.runs(),
      ]);
      setApiStatus(health.status);
      setProducts(productPayload.products);
      setJobs(jobPayload.jobs);
      setDatasets(datasetPayload.datasets);
      setRuns(runPayload.runs);
      if (!selectedProductId && productPayload.products.length > 0) {
        setSelectedProductId(productPayload.products[0].product_id);
      }
      setLoadState({ loading: false, message: "Data refreshed", error: null });
    } catch (error) {
      setApiStatus("error");
      setLoadState({ loading: false, message: null, error: error instanceof Error ? error.message : "Refresh failed" });
    }
  }, [selectedProductId]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  useEffect(() => {
    if (!selectedProduct?.product_id) {
      setProductDetail(null);
      return;
    }
    let active = true;
    api
      .product(selectedProduct.product_id)
      .then((detail) => {
        if (active) setProductDetail(detail);
      })
      .catch(() => {
        if (active) setProductDetail(null);
      });
    return () => {
      active = false;
    };
  }, [selectedProduct?.product_id]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Database size={22} />
          <div>
            <strong>Buying Assistant</strong>
            <span>Production RAG Console</span>
          </div>
        </div>

        <nav className="nav">
          {views.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={view === item.id ? "active" : ""}
                onClick={() => setView(item.id)}
                type="button"
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-status">
          <StatusPill status={apiStatus} />
          <button className="icon-command" onClick={() => void refreshAll()} type="button" title="Refresh">
            <RefreshCw size={16} />
          </button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">Online Buying Assistant</span>
            <h1>{views.find((item) => item.id === view)?.label}</h1>
          </div>
          <Notice state={loadState} />
        </header>

        {view === "library" && (
          <LibraryView
            products={products}
            selectedProductId={selectedProduct?.product_id ?? ""}
            onSelect={setSelectedProductId}
            detail={productDetail}
          />
        )}
        {view === "ingestion" && <IngestionView jobs={jobs} onChanged={refreshAll} />}
        {view === "ask" && (
          <AskView
            products={products}
            selectedProductId={selectedProduct?.product_id ?? ""}
            onSelect={setSelectedProductId}
            answer={answer}
            setAnswer={setAnswer}
          />
        )}
        {view === "evals" && <EvalView datasets={datasets} runs={runs} onChanged={refreshAll} />}
      </section>
    </main>
  );
}

function LibraryView({
  products,
  selectedProductId,
  onSelect,
  detail,
}: {
  products: ProductSummary[];
  selectedProductId: string;
  onSelect: (productId: string) => void;
  detail: ProductDetail | null;
}) {
  return (
    <div className="content-grid two">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Products</span>
            <h2>Catalog</h2>
          </div>
          <Search size={18} />
        </div>
        {products.length === 0 ? (
          <EmptyState title="No products found" detail="Create an ingestion job after starting the backend services." />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Product</th>
                  <th>Corpus</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {products.map((product) => (
                  <tr
                    key={product.id}
                    className={product.product_id === selectedProductId ? "selected" : ""}
                    onClick={() => onSelect(product.product_id)}
                  >
                    <td>
                      <strong>{product.title}</strong>
                      <span>{product.product_id}</span>
                    </td>
                    <td>
                      <span>{product.counts.reviews ?? 0} reviews</span>
                      <span>{product.counts.chunks ?? 0} chunks</span>
                    </td>
                    <td>{formatDate(product.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Selected Product</span>
            <h2>{detail?.title ?? "Details"}</h2>
          </div>
          <PackageSearch size={18} />
        </div>
        {!detail ? (
          <EmptyState title="No product selected" detail="Choose a product from the catalog." />
        ) : (
          <>
            <div className="metrics">
              <Metric label="Sections" value={detail.counts.sections ?? detail.sections.length} />
              <Metric label="Reviews" value={detail.counts.reviews ?? detail.reviews.length} />
              <Metric label="Chunks" value={detail.counts.chunks ?? detail.chunks.length} />
            </div>
            <div className="detail-stack">
              <div>
                <h3>Recent Reviews</h3>
                <div className="rows">
                  {detail.reviews.slice(0, 5).map((review) => (
                    <article className="row-card" key={review.id}>
                      <div>
                        <strong>{review.title || review.source_id}</strong>
                        <span>
                          {review.rating ?? "?"}/5 · {review.date ?? "undated"} · {review.helpful_votes} helpful
                        </span>
                      </div>
                      <p>{review.body}</p>
                    </article>
                  ))}
                </div>
              </div>
              <div>
                <h3>Page Sections</h3>
                <div className="rows">
                  {detail.sections.slice(0, 4).map((section) => (
                    <article className="row-card compact" key={section.id}>
                      <strong>{section.section_type}</strong>
                      <p>{section.text}</p>
                    </article>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function IngestionView({ jobs, onChanged }: { jobs: IngestionJob[]; onChanged: () => Promise<void> }) {
  const [productHtml, setProductHtml] = useState("");
  const [reviewsFile, setReviewsFile] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [enqueue, setEnqueue] = useState(true);
  const [state, setState] = useState<LoadState>(emptyLoadState);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setState({ loading: true, message: null, error: null });
    try {
      await api.createJob({
        product_html: productHtml.trim(),
        reviews_file: reviewsFile.trim(),
        source_url: sourceUrl.trim() || undefined,
        enqueue,
      });
      setProductHtml("");
      setReviewsFile("");
      setSourceUrl("");
      setState({ loading: false, message: "Ingestion job created", error: null });
      await onChanged();
    } catch (error) {
      setState({ loading: false, message: null, error: error instanceof Error ? error.message : "Job creation failed" });
    }
  }

  return (
    <div className="content-grid two">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Raw Files</span>
            <h2>Create Job</h2>
          </div>
          <UploadCloud size={18} />
        </div>
        <form className="form" onSubmit={submit}>
          <label>
            Product HTML
            <input value={productHtml} onChange={(event) => setProductHtml(event.target.value)} placeholder="pan.html" required />
          </label>
          <label>
            Reviews CSV/JSONL
            <input value={reviewsFile} onChange={(event) => setReviewsFile(event.target.value)} placeholder="pan_reviews.csv" required />
          </label>
          <label>
            Source URL
            <input value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder="optional" />
          </label>
          <label className="inline-check">
            <input checked={enqueue} onChange={(event) => setEnqueue(event.target.checked)} type="checkbox" />
            Queue worker execution
          </label>
          <button className="primary-command" type="submit" disabled={state.loading}>
            {state.loading ? <Loader2 className="spin" size={16} /> : <UploadCloud size={16} />}
            <span>Create Job</span>
          </button>
        </form>
        <Notice state={state} />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Pipeline</span>
            <h2>Ingestion Jobs</h2>
          </div>
          <Clock3 size={18} />
        </div>
        {jobs.length === 0 ? (
          <EmptyState title="No ingestion jobs" detail="Create a job from raw product files." />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Input</th>
                  <th>Stats</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td>
                      <StatusPill status={job.status} />
                    </td>
                    <td>
                      <strong>{job.input_uri ?? "n/a"}</strong>
                      <span>{job.job_id}</span>
                    </td>
                    <td>
                      <span>{compactJson(job.stats)}</span>
                    </td>
                    <td>{formatDate(job.completed_at ?? job.started_at ?? job.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function AskView({
  products,
  selectedProductId,
  onSelect,
  answer,
  setAnswer,
}: {
  products: ProductSummary[];
  selectedProductId: string;
  onSelect: (productId: string) => void;
  answer: AskResponse | null;
  setAnswer: (answer: AskResponse | null) => void;
}) {
  const [question, setQuestion] = useState("");
  const [model, setModel] = useState("extractive");
  const [state, setState] = useState<LoadState>(emptyLoadState);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedProductId) return;
    setState({ loading: true, message: null, error: null });
    try {
      const response = await api.ask({
        product_id: selectedProductId,
        question: question.trim(),
        model,
      });
      setAnswer(response);
      setState({ loading: false, message: "Answer generated", error: null });
    } catch (error) {
      setState({ loading: false, message: null, error: error instanceof Error ? error.message : "Ask failed" });
    }
  }

  return (
    <div className="content-grid ask-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Question</span>
            <h2>Ask Product Evidence</h2>
          </div>
          <MessageSquare size={18} />
        </div>
        <form className="form" onSubmit={submit}>
          <label>
            Product
            <select value={selectedProductId} onChange={(event) => onSelect(event.target.value)} required>
              {products.map((product) => (
                <option value={product.product_id} key={product.id}>
                  {product.title}
                </option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              <option value="extractive">extractive</option>
              <option value="llama3.1:8b">llama3.1:8b</option>
              <option value="mistral:7b">mistral:7b</option>
              <option value="qwen2:7b">qwen2:7b</option>
            </select>
          </label>
          <label>
            Question
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="How long does the coating last?"
              rows={5}
              required
            />
          </label>
          <button className="primary-command" type="submit" disabled={!products.length || state.loading}>
            {state.loading ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
            <span>Ask</span>
          </button>
        </form>
        <Notice state={state} />
      </section>

      <section className="panel wide">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Answer</span>
            <h2>{answer ? answer.result.engine : "Evidence Response"}</h2>
          </div>
          <BarChart3 size={18} />
        </div>
        {!answer ? (
          <EmptyState title="No answer yet" detail="Ask a question to inspect evidence, diagnostics, and temporal signals." />
        ) : (
          <div className="answer-grid">
            <div className="answer-main">
              <p className="answer-text">{answer.result.answer}</p>
              <div className="metrics">
                <Metric label="Confidence" value={answer.result.confidence.toFixed(2)} />
                <Metric label="Evidence" value={answer.evidence.length} />
                <Metric label="Latency" value={`${answer.diagnostics.latency_ms ?? 0}ms`} />
              </div>
              <h3>Evidence</h3>
              <div className="rows">
                {answer.evidence.map((item) => (
                  <article className="row-card" key={`${item.source_id}-${item.score}`}>
                    <div>
                      <strong>
                        [{item.source_id}] {item.source_type}
                      </strong>
                      <span>score {item.score.toFixed(3)}</span>
                    </div>
                    <p>{item.text}</p>
                    {item.score_components && <pre>{compactJson(item.score_components)}</pre>}
                  </article>
                ))}
              </div>
            </div>
            <aside className="diagnostics">
              <h3>Diagnostics</h3>
              <pre>{compactJson(answer.diagnostics)}</pre>
              <h3>Faithfulness</h3>
              <pre>{compactJson(answer.faithfulness)}</pre>
              <h3>Temporal</h3>
              <pre>{compactJson(answer.consensus)}</pre>
            </aside>
          </div>
        )}
      </section>
    </div>
  );
}

function EvalView({ datasets, runs, onChanged }: { datasets: EvalDataset[]; runs: EvalRun[]; onChanged: () => Promise<void> }) {
  const [datasetName, setDatasetName] = useState("groundtruth");
  const [sourcePath, setSourcePath] = useState("eval/groundtruth_dataset.jsonl");
  const [replace, setReplace] = useState(true);
  const [runDataset, setRunDataset] = useState("");
  const [model, setModel] = useState("extractive");
  const [k, setK] = useState(5);
  const [state, setState] = useState<LoadState>(emptyLoadState);

  useEffect(() => {
    if (!runDataset && datasets.length > 0) {
      setRunDataset(datasets[0].name);
    }
  }, [datasets, runDataset]);

  async function createDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setState({ loading: true, message: null, error: null });
    try {
      await api.createDataset({
        name: datasetName.trim(),
        source_path: sourcePath.trim(),
        replace,
      });
      setState({ loading: false, message: "Eval dataset saved", error: null });
      await onChanged();
    } catch (error) {
      setState({ loading: false, message: null, error: error instanceof Error ? error.message : "Dataset creation failed" });
    }
  }

  async function createRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setState({ loading: true, message: null, error: null });
    try {
      await api.createRun({ dataset_name: runDataset, model, k });
      setState({ loading: false, message: "Eval run completed", error: null });
      await onChanged();
    } catch (error) {
      setState({ loading: false, message: null, error: error instanceof Error ? error.message : "Eval run failed" });
    }
  }

  return (
    <div className="content-grid two">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Datasets</span>
            <h2>Manage Ground Truth</h2>
          </div>
          <FlaskConical size={18} />
        </div>
        <form className="form" onSubmit={createDataset}>
          <label>
            Name
            <input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} required />
          </label>
          <label>
            Source JSONL
            <input value={sourcePath} onChange={(event) => setSourcePath(event.target.value)} required />
          </label>
          <label className="inline-check">
            <input checked={replace} onChange={(event) => setReplace(event.target.checked)} type="checkbox" />
            Replace existing dataset
          </label>
          <button className="primary-command" type="submit" disabled={state.loading}>
            <UploadCloud size={16} />
            <span>Save Dataset</span>
          </button>
        </form>

        <form className="form separated" onSubmit={createRun}>
          <label>
            Dataset
            <select value={runDataset} onChange={(event) => setRunDataset(event.target.value)} required>
              {datasets.map((dataset) => (
                <option value={dataset.name} key={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              <option value="extractive">extractive</option>
              <option value="llama3.1:8b">llama3.1:8b</option>
            </select>
          </label>
          <label>
            k
            <input value={k} onChange={(event) => setK(Number(event.target.value))} type="number" min={1} max={20} />
          </label>
          <button className="primary-command" type="submit" disabled={!datasets.length || state.loading}>
            <Play size={16} />
            <span>Run Eval</span>
          </button>
        </form>
        <Notice state={state} />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Evaluation</span>
            <h2>Runs</h2>
          </div>
          <BarChart3 size={18} />
        </div>
        <div className="metrics">
          <Metric label="Datasets" value={datasets.length} />
          <Metric label="Runs" value={runs.length} />
          <Metric label="Succeeded" value={runs.filter((run) => run.status === "succeeded").length} />
        </div>
        {runs.length === 0 ? (
          <EmptyState title="No eval runs" detail="Save a dataset, then run the extractive baseline." />
        ) : (
          <div className="rows">
            {runs.map((run) => (
              <article className="row-card" key={run.id}>
                <div>
                  <strong>{run.model}</strong>
                  <StatusPill status={run.status} />
                </div>
                <span>{formatDate(run.completed_at ?? run.created_at)}</span>
                <pre>{compactJson(run.metrics)}</pre>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
