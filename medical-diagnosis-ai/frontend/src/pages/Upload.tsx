import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { api } from "../services/api";
import { Spinner } from "../components/Spinner";
import { assetUrl } from "../lib/url";

type PredictResponse = {
  study_id: number;
  modality: string;
  labels: Record<string, number>;
  top_findings: string[];
  heatmap_url: string | null;
  mask_url: string | null;
  severity_score: number;
  prediction_id: number;
};

export default function Upload() {
  const [modality, setModality] = useState<"chest_xray" | "brain_mri">("chest_xray");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [heatmapBlob, setHeatmapBlob] = useState<string | null>(null);
  const [maskBlob, setMaskBlob] = useState<string | null>(null);

  useEffect(() => {
    if (!result?.heatmap_url) {
      setHeatmapBlob(null);
      return;
    }
    let cancelled = false;
    let url: string | null = null;
    void (async () => {
      try {
        const res = await api.get(assetUrl(result.heatmap_url), { responseType: "blob" });
        url = URL.createObjectURL(res.data);
        if (!cancelled) setHeatmapBlob(url);
      } catch {
        if (!cancelled) setHeatmapBlob(null);
      }
    })();
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [result?.heatmap_url]);

  useEffect(() => {
    if (!result?.mask_url) {
      setMaskBlob(null);
      return;
    }
    let cancelled = false;
    let url: string | null = null;
    void (async () => {
      try {
        const res = await api.get(assetUrl(result.mask_url), { responseType: "blob" });
        url = URL.createObjectURL(res.data);
        if (!cancelled) setMaskBlob(url);
      } catch {
        if (!cancelled) setMaskBlob(null);
      }
    })();
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [result?.mask_url]);

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted[0]) setFile(accepted[0]);
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg"], "application/dicom": [".dcm", ".dicom"] },
    multiple: false,
  });

  async function runPipeline() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setReport(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("modality", modality);
      const up = await api.post<{ study_id: number }>("/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const pred = await api.post<PredictResponse>("/predict", {
        study_id: up.data.study_id,
        modality,
        run_segmentation: true,
        run_heatmap: true,
      });
      setResult(pred.data);
      const rep = await api.post<{ findings: string; impression: string; suggested_action: string }>(
        "/generate-report",
        { prediction_id: pred.data.prediction_id, use_llm: false },
      );
      setReport(`${rep.data.findings}\n\n${rep.data.impression}\n\n${rep.data.suggested_action}`);
    } catch (e) {
      setError("Upload or inference failed. Ensure the API is running and you are signed in.");
    } finally {
      setBusy(false);
    }
  }

  function speakReport() {
    if (!report) return;
    const u = new SpeechSynthesisUtterance(report);
    u.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }

  return (
    <div className="mx-auto max-w-5xl space-y-8 px-4 py-10">
      <div>
        <h1 className="text-3xl font-bold">Analyze scan</h1>
        <p className="text-slate-500 dark:text-slate-400">Chest X-ray (multi-label) or brain MRI (4-class)</p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-4">
          <label className="block text-sm font-medium">
            Modality
            <select
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
              value={modality}
              onChange={(e) => setModality(e.target.value as "chest_xray" | "brain_mri")}
            >
              <option value="chest_xray">Chest X-ray</option>
              <option value="brain_mri">Brain MRI</option>
            </select>
          </label>

          <div
            {...getRootProps()}
            className={`flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition ${
              isDragActive
                ? "border-brand-500 bg-brand-50 dark:bg-slate-900"
                : "border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900"
            }`}
          >
            <input {...getInputProps()} />
            <p className="font-medium">Drag & drop an image here</p>
            <p className="mt-1 text-xs text-slate-500">PNG, JPG, or DICOM</p>
            {file ? <p className="mt-3 text-sm text-brand-600">Selected: {file.name}</p> : null}
          </div>

          <button
            type="button"
            disabled={!file || busy}
            onClick={() => void runPipeline()}
            className="w-full rounded-xl bg-brand-600 py-3 text-sm font-semibold text-white hover:bg-brand-500 disabled:opacity-50"
          >
            {busy ? "Running AI pipeline…" : "Upload & predict"}
          </button>
          {error ? <p className="text-sm text-red-600">{error}</p> : null}
        </div>

        <div className="space-y-4">
          {busy ? <Spinner label="Loading model, Grad-CAM, segmentation…" /> : null}
          {result ? (
            <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h2 className="text-lg font-semibold">Predictions</h2>
              <ul className="space-y-1 text-sm">
                {Object.entries(result.labels)
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => (
                    <li key={k} className="flex justify-between gap-4">
                      <span className="capitalize">{k.replace(/_/g, " ")}</span>
                      <span className="font-mono text-brand-600">{(v * 100).toFixed(1)}%</span>
                    </li>
                  ))}
              </ul>
              <p className="text-xs text-slate-500">Severity score: {(result.severity_score * 100).toFixed(1)} / 100 (scaled)</p>
              <div className="grid gap-3 sm:grid-cols-2">
                {heatmapBlob ? (
                  <div>
                    <p className="mb-1 text-xs font-semibold uppercase text-slate-500">Grad-CAM</p>
                    <img
                      src={heatmapBlob}
                      alt="Grad-CAM heatmap"
                      className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                    />
                  </div>
                ) : null}
                {maskBlob ? (
                  <div>
                    <p className="mb-1 text-xs font-semibold uppercase text-slate-500">Segmentation</p>
                    <img
                      src={maskBlob}
                      alt="Segmentation overlay"
                      className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                    />
                  </div>
                ) : null}
              </div>
              {report ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="text-sm font-semibold">Clinical-style report</h3>
                    <button
                      type="button"
                      onClick={speakReport}
                      className="rounded-lg border border-slate-200 px-2 py-1 text-xs dark:border-slate-700"
                    >
                      Read aloud
                    </button>
                  </div>
                  <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-3 text-xs text-slate-700 dark:bg-slate-950 dark:text-slate-200">
                    {report}
                  </pre>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="text-sm text-slate-500">Results will appear here after inference.</p>
          )}
        </div>
      </div>
    </div>
  );
}
