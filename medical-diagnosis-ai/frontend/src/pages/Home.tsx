import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-8 px-4 py-20 text-center">
      <p className="text-sm font-semibold uppercase tracking-wide text-brand-600">Clinical decision support</p>
      <h1 className="text-4xl font-bold leading-tight md:text-5xl">
        AI Medical Diagnosis Assistant for X-Ray and MRI Analysis
      </h1>
      <p className="text-lg text-slate-600 dark:text-slate-300">
        Multi-label chest screening, brain MRI classification, Grad-CAM explainability, optional U-Net segmentation, and
        structured reporting — behind a secure FastAPI backend and modern React dashboard.
      </p>
      <div className="flex flex-wrap justify-center gap-3">
        <Link
          to="/register"
          className="rounded-xl bg-brand-600 px-6 py-3 text-sm font-semibold text-white shadow hover:bg-brand-500"
        >
          Get started
        </Link>
        <Link
          to="/login"
          className="rounded-xl border border-slate-300 px-6 py-3 text-sm font-semibold text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-100 dark:hover:bg-slate-900"
        >
          Sign in
        </Link>
      </div>
    </div>
  );
}
