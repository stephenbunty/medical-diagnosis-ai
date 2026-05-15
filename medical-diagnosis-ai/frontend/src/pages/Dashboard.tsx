import { useEffect, useState } from "react";
import { api } from "../services/api";
import { ModalityDoughnut, SeverityBar } from "../components/Charts";
import { Spinner } from "../components/Spinner";

type Stats = {
  predictions_by_modality: Record<string, number>;
  average_severity: number;
  total_predictions: number;
};

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      try {
        const { data } = await api.get<Stats>("/stats");
        setStats(data);
      } catch {
        setStats({ predictions_by_modality: {}, average_severity: 0, total_predictions: 0 });
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <Spinner label="Loading analytics…" />;

  const chest = stats?.predictions_by_modality?.chest_xray ?? 0;
  const brain = stats?.predictions_by_modality?.brain_mri ?? 0;

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-4 py-10">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-slate-500 dark:text-slate-400">Prediction volume and severity overview</p>
      </div>
      <div className="grid gap-6 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <p className="text-sm font-medium text-slate-500">Total predictions</p>
          <p className="mt-2 text-4xl font-semibold">{stats?.total_predictions ?? 0}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 md:col-span-2">
          <p className="text-sm font-medium text-slate-500">Average model severity score</p>
          <SeverityBar value={stats?.average_severity ?? 0} />
        </div>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h2 className="mb-4 text-lg font-semibold">Modality mix</h2>
          <ModalityDoughnut chest={chest} brain={brain} />
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h2 className="mb-2 text-lg font-semibold">Workflow</h2>
          <ol className="list-decimal space-y-2 pl-5 text-sm text-slate-600 dark:text-slate-300">
            <li>Upload a chest X-ray or brain MRI (PNG, JPG, or DICOM).</li>
            <li>Run AI prediction with Grad-CAM and optional segmentation.</li>
            <li>Generate a structured report and use text-to-speech in the browser if desired.</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
