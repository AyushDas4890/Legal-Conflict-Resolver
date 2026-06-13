'use client'
import { useRef, useEffect, useState } from 'react'
import { motion, useMotionValue, useTransform, AnimatePresence } from 'framer-motion'
import { ArrowRight, Scales, Warning, CheckCircle, Spinner } from '@phosphor-icons/react'

/* ── variants ─────────────────────────────────────────── */
const EASE = [0.16, 1, 0.3, 1]

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
}
const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.65, ease: EASE } },
}

/* ── demo card data ───────────────────────────────────── */
const STAGES = ['scanning', 'results']

const DEMO_CONFLICTS = [
  { label: 'CONTRADICTION', severity: 'HIGH',   clause: 'Limitation of Liability', prob: 0.94 },
  { label: 'CONTRADICTION', severity: 'MEDIUM', clause: 'Indemnification',          prob: 0.81 },
  { label: 'ENTAILMENT',    severity: 'LOW',    clause: 'Governing Law',             prob: 0.63 },
]

const SEV_COLORS = {
  HIGH:   'text-red-400 bg-red-400/10 border-red-400/20',
  MEDIUM: 'text-amber-400 bg-amber-400/10 border-amber-400/20',
  LOW:    'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
}

/* ── magnetic button ──────────────────────────────────── */
function MagneticButton({ children, className, href }) {
  const ref = useRef(null)
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const tx = useTransform(x, (v) => `${v * 0.35}px`)
  const ty = useTransform(y, (v) => `${v * 0.35}px`)

  const handleMouseMove = (e) => {
    const rect = ref.current.getBoundingClientRect()
    x.set(e.clientX - rect.left - rect.width / 2)
    y.set(e.clientY - rect.top - rect.height / 2)
  }
  const handleMouseLeave = () => { x.set(0); y.set(0) }

  return (
    <motion.a
      ref={ref}
      href={href}
      style={{ x: tx, y: ty }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={className}
      whileTap={{ scale: 0.97 }}
    >
      {children}
    </motion.a>
  )
}

/* ── animated live card ───────────────────────────────── */
function LiveCard() {
  const [stage, setStage] = useState('scanning')
  const [scanPct, setScanPct] = useState(0)

  useEffect(() => {
    let raf
    let start = null
    const DURATION = 1800 // ms for scan to fill

    const tick = (ts) => {
      if (!start) start = ts
      const elapsed = ts - start
      const pct = Math.min((elapsed / DURATION) * 100, 100)
      setScanPct(pct)
      if (pct < 100) {
        raf = requestAnimationFrame(tick)
      } else {
        setTimeout(() => {
          setStage('results')
          setTimeout(() => {
            setStage('scanning')
            setScanPct(0)
            start = null
            raf = requestAnimationFrame(tick)
          }, 3400)
        }, 300)
      }
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0, y: 28, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.8, ease: EASE, delay: 0.45 }}
      className="relative rounded-xl border border-zinc-800 bg-zinc-900/70 backdrop-blur-sm p-5 shadow-[0_0_70px_-20px_rgba(245,158,11,0.18)]"
    >
      {/* header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Scales size={14} weight="fill" className="text-amber-500" />
          <span className="text-xs font-medium text-zinc-300 tracking-tight">Conflict Analysis</span>
        </div>
        <AnimatePresence mode="wait">
          {stage === 'scanning' ? (
            <motion.span
              key="scanning"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 text-[10px] font-mono text-amber-400/80 bg-zinc-800 rounded px-2 py-0.5"
            >
              <Spinner size={10} className="animate-spin" />
              scanning…
            </motion.span>
          ) : (
            <motion.span
              key="done"
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="text-[10px] font-mono text-emerald-400 bg-zinc-800 rounded px-2 py-0.5"
            >
              3 found
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* doc labels */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        {['contract_a.pdf', 'contract_b.pdf'].map((name, i) => (
          <div key={name} className="flex items-center gap-1.5 bg-zinc-800/60 rounded-md px-2.5 py-1.5 border border-zinc-700/50">
            <span className="text-[10px] font-mono text-zinc-500">Doc {i + 1}</span>
            <span className="text-[10px] text-zinc-400 truncate">{name}</span>
          </div>
        ))}
      </div>

      {/* scanning bar */}
      <AnimatePresence mode="wait">
        {stage === 'scanning' ? (
          <motion.div key="bar" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mb-3">
            <div className="flex justify-between mb-1">
              <span className="text-[10px] font-mono text-zinc-600">cross-comparing clauses</span>
              <span className="text-[10px] font-mono text-zinc-500">{Math.round(scanPct)}%</span>
            </div>
            <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-amber-500 rounded-full"
                style={{ width: `${scanPct}%` }}
              />
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="rows"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {DEMO_CONFLICTS.map((cf, i) => (
              <motion.div
                key={cf.clause}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1, duration: 0.4, ease: EASE }}
                className="flex items-center justify-between gap-3 py-2.5 border-b border-zinc-800/60 last:border-0"
              >
                <div className="flex items-center gap-2 min-w-0">
                  {cf.label === 'CONTRADICTION'
                    ? <Warning size={13} weight="fill" className="text-amber-500 shrink-0" />
                    : <CheckCircle size={13} weight="fill" className="text-emerald-500 shrink-0" />
                  }
                  <span className="text-xs text-zinc-300 truncate font-mono">{cf.clause}</span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-xs font-mono text-zinc-500">{(cf.prob * 100).toFixed(0)}%</span>
                  <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${SEV_COLORS[cf.severity]}`}>
                    {cf.severity}
                  </span>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* scan line */}
      <div className="mt-3 h-px bg-gradient-to-r from-transparent via-amber-500/40 to-transparent animate-scan" />
      <p className="text-[10px] font-mono text-zinc-600 mt-2 text-center">
        cross_compare · nli_model · token_attribution
      </p>

      {/* ambient glow */}
      <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 w-48 h-16 bg-amber-500/10 blur-2xl pointer-events-none rounded-full" />
    </motion.div>
  )
}

/* ── main hero ────────────────────────────────────────── */
export default function Hero() {
  return (
    <section id="hero" className="relative min-h-[100dvh] flex items-center overflow-hidden">

      {/* grid bg */}
      <div
        className="absolute inset-0 opacity-[0.025]"
        style={{
          backgroundImage:
            'linear-gradient(#f5a623 1px,transparent 1px),linear-gradient(90deg,#f5a623 1px,transparent 1px)',
          backgroundSize: '44px 44px',
        }}
      />

      {/* glow blob */}
      <motion.div
        className="absolute top-1/4 left-1/3 w-[700px] h-[700px] rounded-full bg-amber-500/6 blur-[140px] pointer-events-none"
        animate={{ scale: [1, 1.08, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 8, ease: 'easeInOut', repeat: Infinity }}
      />

      <div className="relative max-w-7xl mx-auto px-6 pt-24 pb-16 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-[55fr_45fr] gap-16 items-center">

          {/* ── left ── */}
          <motion.div variants={container} initial="hidden" animate="show">

            {/* badge */}
            <motion.div variants={item} className="inline-flex items-center gap-2 text-[11px] font-mono text-amber-400/80 bg-amber-400/8 border border-amber-400/20 rounded-full px-3 py-1 mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse-slow" />
              DeBERTa-v3-large · NLI Pipeline
            </motion.div>

            {/* h1 */}
            <motion.h1
              variants={item}
              className="text-5xl md:text-[4.5rem] font-bold tracking-tighter leading-[1.04] text-zinc-50 mb-6"
              style={{ textWrap: 'balance' }}
            >
              Find what your<br />
              contracts{' '}
              <span className="text-amber-400">disagree</span>{' '}
              on.
            </motion.h1>

            {/* desc */}
            <motion.p variants={item} className="text-zinc-400 text-base leading-relaxed max-w-[52ch] mb-10">
              Upload two legal documents. Our NLI model cross-compares every clause pair, surfaces contradictions by severity, and returns token-level evidence — so you know exactly where the conflict lives.
            </motion.p>

            {/* CTAs */}
            <motion.div variants={item} className="flex flex-wrap items-center gap-3">
              <MagneticButton
                href="#analyzer"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-amber-500 hover:bg-amber-400 text-zinc-950 text-sm font-semibold transition-colors duration-200 cursor-pointer select-none"
              >
                Analyze documents
                <ArrowRight size={14} weight="bold" />
              </MagneticButton>

              <MagneticButton
                href="https://github.com/AyushDas4890/Legal-Conflict-Resolver"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-zinc-800/80 hover:bg-zinc-800 text-zinc-200 text-sm font-medium border border-zinc-700 transition-colors duration-200 cursor-pointer select-none"
              >
                View on GitHub
              </MagneticButton>
            </motion.div>

            {/* stats */}
            <motion.div variants={item} className="flex flex-wrap gap-8 mt-12">
              {[
                { value: '87.0%', label: 'Manual review reduction' },
                { value: '94.2%', label: 'NLI precision' },
                { value: '< 3s',  label: 'Per comparison' },
              ].map(({ value, label }) => (
                <div key={label}>
                  <div className="text-xl font-bold font-mono text-zinc-100 tracking-tight">{value}</div>
                  <div className="text-xs text-zinc-500 mt-0.5">{label}</div>
                </div>
              ))}
            </motion.div>

          </motion.div>

          {/* ── right ── */}
          <div className="hidden lg:block">
            <LiveCard />
          </div>

        </div>
      </div>
    </section>
  )
}
