export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center gap-3 py-10">
      <div className="h-10 w-10 animate-spin rounded-full border-4 border-slate-200 border-t-brand-600 dark:border-slate-700 dark:border-t-brand-500" />
      {label ? <p className="text-sm text-slate-500">{label}</p> : null}
    </div>
  );
}
