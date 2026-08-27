import { useEffect, useRef, type ReactNode } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

/** Scroll-triggered fade-up reveal. Children stagger if `stagger` is set. */
export function Reveal({
  children,
  className,
  stagger = 0,
  y = 18,
}: {
  children: ReactNode;
  className?: string;
  stagger?: number;
  y?: number;
}): ReactNode {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const targets = stagger > 0 ? Array.from(el.children) : el;
    const tween = gsap.fromTo(
      targets,
      { autoAlpha: 0, y },
      {
        autoAlpha: 1,
        y: 0,
        duration: 0.7,
        ease: 'power2.out',
        stagger,
        scrollTrigger: { trigger: el, start: 'top 85%', once: true },
      },
    );
    return () => {
      tween.scrollTrigger?.kill();
      tween.kill();
    };
  }, [stagger, y]);
  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

/** GSAP number count-up. Animates from 0 to `value` when it scrolls into view. */
export function CountUp({
  value,
  decimals = 3,
  className,
  duration = 1.2,
}: {
  value: number;
  decimals?: number;
  className?: string;
  duration?: number;
}): ReactNode {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obj = { v: 0 };
    const tween = gsap.to(obj, {
      v: value,
      duration,
      ease: 'power2.out',
      scrollTrigger: { trigger: el, start: 'top 92%', once: true },
      onUpdate: () => {
        el.textContent = obj.v.toFixed(decimals);
      },
    });
    return () => {
      tween.scrollTrigger?.kill();
      tween.kill();
    };
  }, [value, decimals, duration]);
  return (
    <span ref={ref} className={className}>
      {(0).toFixed(decimals)}
    </span>
  );
}
