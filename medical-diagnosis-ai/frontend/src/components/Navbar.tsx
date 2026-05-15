import { Link, NavLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";

const linkCls =
  "px-3 py-2 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800";

export function Navbar() {
  const { user, logout } = useAuth();
  const { theme, toggle } = useTheme();

  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <Link to="/" className="text-lg font-semibold text-brand-600 dark:text-brand-500">
          MedAI Diagnosis
        </Link>
        <nav className="flex flex-wrap items-center gap-1">
          <NavLink to="/dashboard" className={({ isActive }) => linkCls + (isActive ? " bg-slate-100 dark:bg-slate-800" : "")}>
            Dashboard
          </NavLink>
          <NavLink to="/upload" className={({ isActive }) => linkCls + (isActive ? " bg-slate-100 dark:bg-slate-800" : "")}>
            Analyze
          </NavLink>
          <NavLink to="/history" className={({ isActive }) => linkCls + (isActive ? " bg-slate-100 dark:bg-slate-800" : "")}>
            History
          </NavLink>
        </nav>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={toggle}
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium dark:border-slate-700"
            title="Toggle theme"
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          {user ? (
            <div className="flex items-center gap-2 text-sm">
              <span className="hidden sm:inline text-slate-500">{user.email}</span>
              <button
                type="button"
                onClick={logout}
                className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white dark:bg-white dark:text-slate-900"
              >
                Log out
              </button>
            </div>
          ) : (
            <NavLink
              to="/login"
              className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-500"
            >
              Sign in
            </NavLink>
          )}
        </div>
      </div>
    </header>
  );
}
