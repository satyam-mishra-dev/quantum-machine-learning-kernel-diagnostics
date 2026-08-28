import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react';
import clsx from 'clsx';

// Themed primitives inherited from the Tally design system: every color is a
// token, radius 4px, flat 1px --rule borders, no card shadows.

type Variant = 'primary' | 'outline' | 'quantum';

const VARIANT: Record<Variant, string> = {
  primary: 'bg-action text-paper border border-action hover:opacity-90',
  outline: 'bg-paper text-ink border border-rule hover:bg-wash',
  quantum: 'bg-quantum text-paper border border-quantum hover:opacity-90',
};

export function buttonClass(variant: Variant = 'outline', className?: string): string {
  return clsx(
    'inline-flex h-9 items-center justify-center gap-1.5 rounded-sm px-3.5 text-[13px] font-medium whitespace-nowrap',
    'transition-[background-color,color,opacity] duration-150 select-none cursor-pointer',
    'disabled:pointer-events-none disabled:opacity-45',
    VARIANT[variant],
    className,
  );
}

export function Button({
  variant = 'outline',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }): ReactNode {
  return <button className={buttonClass(variant, className)} {...props} />;
}

export type Tone = 'credit' | 'debit' | 'hold' | 'action' | 'quantum' | 'neutral';
const TONE: Record<Tone, string> = {
  credit: 'text-credit border-credit/35 bg-credit-wash',
  debit: 'text-debit border-debit/35 bg-debit-wash',
  hold: 'text-hold border-hold/40 bg-hold-wash',
  action: 'text-action border-action/35 bg-action-wash',
  quantum: 'text-quantum border-quantum/35 bg-quantum-wash',
  neutral: 'text-ink-60 border-rule bg-wash',
};

export function Badge({
  tone = 'neutral',
  className,
  children,
}: {
  tone?: Tone;
  className?: string;
  children: ReactNode;
}): ReactNode {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 font-mono text-[11px] leading-none tnum',
        TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>): ReactNode {
  return <div className={clsx('card', className)} {...props} />;
}

export function PageTitle({ children }: { children: ReactNode }): ReactNode {
  return (
    <h1 className="font-serif text-[26px] leading-tight font-semibold text-ink sm:text-[30px]">
      {children}
    </h1>
  );
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }): ReactNode {
  return <div className={clsx('eyebrow', className)}>{children}</div>;
}

export function Skeleton({ className }: { className?: string }): ReactNode {
  return <div className={clsx('animate-pulse rounded-sm bg-wash', className)} />;
}

export function ErrorState({ children }: { children: ReactNode }): ReactNode {
  return (
    <div className="rounded-sm border border-debit/35 bg-debit-wash px-4 py-3 text-[13px] text-debit">
      {children}
    </div>
  );
}

/** Key–value row used across scout panels / metadata blocks. */
export function KV({ k, v }: { k: ReactNode; v: ReactNode }): ReactNode {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-rule py-2 last:border-b-0">
      <span className="text-[13px] text-ink-60">{k}</span>
      <span className="font-mono text-[13px] text-ink tnum">{v}</span>
    </div>
  );
}

/** Render free tier sleeps after 15 min idle and wakes in ~50s. Say so
 *  wherever a page waits on the API, so a cold start doesn't read as broken. */
export function ColdStartNote({ children }: { children?: ReactNode }): ReactNode {
  return (
    <p className="mt-3 text-[12px] leading-relaxed text-ink/55">
      {children ?? (
        <>
          The backend runs on Render's free tier — if it has been idle it needs
          about a minute to wake up, so the first request can be slow. Later
          ones are quick.
        </>
      )}
    </p>
  );
}
