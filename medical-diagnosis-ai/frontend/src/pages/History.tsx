import { useEffect, useState } from "react";
import { api } from "../services/api";
import { Spinner } from "../components/Spinner";

type Row = {
  prediction_id: number;
  study_id: number;
  modality: string;
  created_at: string;
  labels: Record<string, number>;
  severity_score: number;
};

export default function History() {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      try {
        const { data } = await api.get<Row[]>("/history");
        setRows(data);
      } catch {
        setRows([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <Spinner label="Loading history…" />;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-10">
      <h1 className="text-3xl font-bold">Upload & prediction history</h1>
      <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase text-slate-500 dark:bg-slate-950 dark:text-slate-400">
            <tr>
              <th className="px-4 py-3">ID</th>
              <th className="px-4 py-3">Modality</th>
              <th className="px-4 py-3">Top label</th>
              <th className="px-4 py-3">Severity</th>
              <th className="px-4 py-3">When</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const top = Object.entries(r.labels).sort((a, b) => b[1] - a[1])[0];
              return (
                <tr key={r.prediction_id} className="border-t border-slate-100 dark:border-slate-800">
                  <td className="px-4 py-3 font-mono text-xs">{r.prediction_id}</td>
                  <td className="px-4 py-3">{r.modality}</td>
                  <td className="px-4 py-3">
                    {top ? `${top[0]} (${(top[1] * 100).toFixed(0)}%)` : "—"}
                  </td>
                  <td className="px-4 py-3">{(r.severity_score * 100).toFixed(0)}</td>
                  <td className="px-4 py-3 text-xs text-slate-500">{r.created_at}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length === 0 ? <p className="p-6 text-sm text-slate-500">No predictions yet.</p> : null}
      </div>
    </div>
  );
}
