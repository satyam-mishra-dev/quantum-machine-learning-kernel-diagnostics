import type { ReactNode } from 'react';
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom';
import clsx from 'clsx';
import { Landing } from './pages/Landing';
import { Dashboard } from './pages/Dashboard';
import { DatasetDetail } from './pages/DatasetDetail';
import { PatientCheck } from './pages/PatientCheck';
import { Upload } from './pages/Upload';

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/dashboard', label: 'Dashboard', end: false },
  { to: '/check', label: 'Patient check', end: false },
  { to: '/upload', label: 'Upload', end: false },
];

function BrandMark(): ReactNode {
  return (
    <svg width="18" height="18" viewBox="0 0 32 32" aria-hidden className="shrink-0">
      <defs>
        <linearGradient id="qmark" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--color-quantum)" />
          <stop offset="1" stopColor="var(--color-quantum-2)" />
        </linearGradient>
      </defs>
      <circle cx="15" cy="15" r="9.5" fill="none" stroke="url(#qmark)" strokeWidth="2.6" />
      <line x1="21.5" y1="21.5" x2="27.5" y2="27.5" stroke="url(#qmark)" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}

function Shell({ children }: { children: ReactNode }): ReactNode {
  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-30 border-b border-rule bg-paper/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-2.5">
          <NavLink to="/" className="flex items-center gap-2 text-ink">
            <BrandMark />
            <span className="font-serif text-[17px] font-semibold tracking-tight">QuantumFloor</span>
          </NavLink>
          <nav className="ml-2 flex items-center gap-0.5" aria-label="Primary">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    'rounded-sm px-2.5 py-1.5 text-[13px] font-medium transition-colors',
                    isActive ? 'bg-wash text-ink' : 'text-ink-60 hover:text-ink',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-8">{children}</main>

      <footer className="border-t border-rule">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-4 font-mono text-[11px] text-ink-45">
          <span>QuantumFloor · honest quantum ML for early disease detection</span>
          <span>Built for Smart India Hackathon 2026 · runs on IBM Quantum hardware</span>
        </div>
      </footer>
    </div>
  );
}

export function App(): ReactNode {
  return (
    <BrowserRouter>
      <Shell>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/datasets/:name" element={<DatasetDetail />} />
          <Route path="/check" element={<PatientCheck />} />
          <Route path="/upload" element={<Upload />} />
        </Routes>
      </Shell>
    </BrowserRouter>
  );
}
