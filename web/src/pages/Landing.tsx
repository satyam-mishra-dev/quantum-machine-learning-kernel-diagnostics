import { useEffect, useRef, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import gsap from 'gsap';
import { buttonClass, Card, Eyebrow } from '../components/ui';
import { Reveal } from '../components/motion';
import { HardwareSection } from '../components/Hardware';

/** Slow-orbiting entangled rings — the only decorative element on the page. */
function Orbits(): ReactNode {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const tweens = [
      gsap.to(el.querySelector('.orbit-a'), { rotation: 360, transformOrigin: '50% 50%', duration: 42, repeat: -1, ease: 'none' }),
      gsap.to(el.querySelector('.orbit-b'), { rotation: -360, transformOrigin: '50% 50%', duration: 58, repeat: -1, ease: 'none' }),
    ];
    return () => tweens.forEach((t) => t.kill());
  }, []);
  return (
    <svg
      ref={ref}
      viewBox="0 0 400 400"
      aria-hidden
      className="pointer-events-none absolute top-1/2 right-[-60px] hidden h-[420px] w-[420px] -translate-y-1/2 opacity-[0.16] lg:block"
    >
      <defs>
        <linearGradient id="orb" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--color-quantum)" />
          <stop offset="1" stopColor="var(--color-quantum-2)" />
        </linearGradient>
      </defs>
      <g className="orbit-a">
        <ellipse cx="200" cy="200" rx="180" ry="70" fill="none" stroke="url(#orb)" strokeWidth="1.5" />
        <circle cx="380" cy="200" r="5" fill="var(--color-quantum)" />
      </g>
      <g className="orbit-b">
        <ellipse cx="200" cy="200" rx="70" ry="180" fill="none" stroke="url(#orb)" strokeWidth="1.5" />
        <circle cx="200" cy="20" r="5" fill="var(--color-quantum-2)" />
      </g>
      <circle cx="200" cy="200" r="10" fill="url(#orb)" />
    </svg>
  );
}

const PILLARS = [
  {
    title: 'Qubit budgeter',
    body: 'Compresses any dataset to a hardware-honest qubit budget with PCA, reporting exactly how much variance survives.',
  },
  {
    title: 'Advantage scout',
    body: 'Kernel target alignment and geometric difference decide whether quantum is even worth trying — before a single circuit runs.',
  },
  {
    title: 'Classical twins',
    body: 'Every quantum model races logistic regression, RBF-SVM and random forest on the same split. The verdict is published either way.',
  },
  {
    title: 'Trust layer',
    body: 'Split conformal prediction with a coverage guarantee, plus per-patient SHAP explanations, on every prediction.',
  },
];

export function Landing(): ReactNode {
  const hero = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = hero.current;
    if (!el) return;
    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
    tl.fromTo('.hero-item', { autoAlpha: 0, y: 24 }, { autoAlpha: 1, y: 0, duration: 0.8, stagger: 0.12 })
      .fromTo('.hero-rule', { scaleX: 0, transformOrigin: 'left' }, { scaleX: 1, duration: 0.7 }, '-=0.5');
    return () => {
      tl.kill();
    };
  }, []);

  return (
    <div>
      <section ref={hero} className="relative overflow-hidden py-14 sm:py-20">
        <Orbits />
        <div className="max-w-2xl">
          <div className="hero-item eyebrow">SIH 2026 · Hybrid quantum–classical ML</div>
          <h1 className="hero-item mt-3 font-serif text-[42px] leading-[1.05] font-semibold text-ink sm:text-[56px]">
            <span className="q-gradient-text">Q-Nidaan</span>
          </h1>
          <div className="hero-rule q-gradient-rule mt-4 w-24" />
          <p className="hero-item mt-5 text-[17px] leading-relaxed text-ink-60 sm:text-lg">
            An honest quantum ML platform for early disease detection on small medical datasets.
            Quantum models race their classical twins on every dataset — and we publish the
            verdict either way.
          </p>
          <div className="hero-item mt-8 flex flex-wrap gap-3">
            <Link to="/dashboard" className={buttonClass('quantum')}>
              Explore the dashboard
            </Link>
            <Link to="/upload" className={buttonClass('outline')}>
              Upload your own dataset
            </Link>
          </div>
        </div>
      </section>

      <section className="py-6">
        <HardwareSection />
      </section>

      <section className="py-10">
        <Eyebrow>The pipeline</Eyebrow>
        <Reveal stagger={0.1} className="mt-4 grid gap-4 sm:grid-cols-2">
          {PILLARS.map((p, i) => (
            <Card key={p.title} className="p-5">
              <div className="font-mono text-[11px] text-quantum tnum">0{i + 1}</div>
              <h3 className="mt-1.5 font-serif text-lg font-semibold text-ink">{p.title}</h3>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-60">{p.body}</p>
            </Card>
          ))}
        </Reveal>
      </section>

      <section className="py-10">
        <Reveal>
          <div className="q-gradient-border p-6 sm:p-8">
            <Eyebrow className="text-quantum">The honesty principle</Eyebrow>
            <p className="mt-3 max-w-3xl font-serif text-xl leading-relaxed text-ink sm:text-2xl">
              Most "quantum ML for healthcare" demos hide the classical baseline. Q-Nidaan trains
              it, publishes it, and tells you when quantum <em>lost</em>. That is the only way a
              quantum advantage claim means anything.
            </p>
          </div>
        </Reveal>
      </section>
    </div>
  );
}
