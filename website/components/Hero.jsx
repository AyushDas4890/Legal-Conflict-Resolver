'use client'
import { ArrowRight, Scales, Warning, CheckCircle } from '@phosphor-icons/react'

const DEMO_CONFLICTS = [
  { label: 'CONTRADICTION', severity: 'HIGH', clause: 'Limitation of Liability', prob: 0.94 },
  { label: 'CONTRADICTION', severity: 'MEDIUM', clause: 'Indemnification', prob: 0.81 },
  { label: 'ENTAILMENT', severity: 'LOW', clause: 'Governing Law', prob: 0.63 },
]

function ConflictRow({ item, delay }) {
  const colors = {
    HIGH: 'text-red-400 bg-red-400/10 border-red-400/20',
    MEDIUM: 'text-amber-400 bg-amber-400/10 border-amber-400/20',
    LOW: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
  }
  const isConflict = item.label === 'CONTRADICTION'

  return (
    <div
      className="flex items-center justify-between gap-3 py-2.5 border-b border-zinc-800/60 last:border-0 fade-up"
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'both' }}
    >
      <div className="flex items-center gap-2 min-w-0">
        {isConflict
          ? <Warning size={13} weight="fill" className="text-amber-500 shrink-0" />
          : <CheckCircle size={13} weight="fill" className="text-emerald-500 shrink-0" />
        }
        <span className="text-xs text-zinc-300 truncate font-mono">{item.clause}</span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-xs font-mono text-zinc-500">{(item.prob * 100).toFixed(0)}%</span>
        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${colors[item.severity]}`}>
          {item.severity}
        </span>
      </div>
    </div>
  )
}

export default function Hero() {
  return (
    <section
      id="hero"
      className="relative min-h-[100dvh] flex items-center overflow-hidden"
    >
      {/* Background grid */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: 'linear-gradient(#f5a623 1px, transparent 1px), linear-gradient(90deg, #f5a623 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      {/* Ambient glow */}
      <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] rounded-full bg-amber-500/5 blur-[120px] pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-6 pt-24 pb-16 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-[55fr_45fr] gap-16 items-center">

          {/* Left — copy */}
          <div>
            <div className="inline-flex items-center gap-2 text-[11px] font-mono text-amber-400/80 bg-amber-400/8 border border-amber-400/20 rounded-full px-3 py-1 mb-8 fade-up">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse-slow" />
              DeBERTa-v3-large · NLI Pipeline
            </div>

            <h1 className="text-4xl md:text-6xl font-bold tracking-tighter leading-[1.05] text-zinc-50 mb-6 fade-up delay-100">
              Find what your<br />
              contracts{' '}
              <span className="text-amber-400">disagree</span>{' '}
              on.
            </h1>

            <p className="text-zinc-400 text-base leading-relaxed max-w-[52ch] mb-10 fade-up delay-200">
              Upload two legal documents. Our NLI model cross-compares every clause pair, surfaces contradictions by severity, and returns token-level evidence — so you know exactly where the conflict lives.
            </p>

            <div className="flex flex-wrap items-center gap-3 fade-up delay-300">
              <a
                href="#analyzer"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-amber-500 hover:bg-amber-400 text-zinc-950 text-sm font-semibold transition-all duration-200 active:scale-[0.97] active:-translate-y-[1px]"
              >
                Analyze documents
                <ArrowRight size={14} weight="bold" />
              </a>
              <a
                href="https://github.com/AyushDas4890/Legal-Conflict-Resolver"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-zinc-800/80 hover:bg-zinc-800 text-zinc-200 text-sm font-medium border border-zinc-700 transition-all duration-200 active:scale-[0.97]"
              >
                View on GitHub
              </a>
            </div>

            {/* Stats row */}
            <div className="flex flex-wrap gap-8 mt-12 fade-up delay-400">
              {[
                { value: '94.2%', label: 'NLI accuracy' },
                { value: '< 3s', label: 'per comparison' },
                { value: 'PDF · DOCX', label: 'supported formats' },
              ].map(({ value, label }) => (
                <div key={label}>
                  <div className="text-xl font-bold font-mono text-zinc-100 tracking-tight">{value}</div>
                  <div className="text-xs text-zinc-500 mt-0.5">{label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right — animated conflict preview */}
          <div className="hidden lg:block">
            <div
              className="relative rounded-xl border border-zinc-800 bg-zinc-900/60 backdrop-blur p-5 shadow-[0_0_60px_-20px_rgba(245,158,11,0.12)] animate-float"
              style={{ animationDelay: '0.5s' }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Scales size={14} weight="fill" className="text-amber-500" />
                  <span className="text-xs font-medium text-zinc-300 tracking-tight">Conflict Analysis</span>
                </div>
                <span className="text-[10px] font-mono text-zinc-500 bg-zinc-800 rounded px-2 py-0.5">3 found</span>
              </div>

              {/* Doc labels */}
              <div className="grid grid-cols-2 gap-2 mb-4">
                {['contract_a.pdf', 'contract_b.pdf'].map((name, i) => (
                  <div key={name} className="flex items-center gap-1.5 bg-zinc-800/60 rounded-md px-2.5 py-1.5 border border-zinc-700/50">
                    <span className="text-[10px] font-mono text-zinc-500">Doc {i + 1}</span>
                    <span className="text-[10px] text-zinc-400 truncate">{name}</span>
                  </div>
                ))}
              </div>

              {/* Conflicts */}
              <div>
                {DEMO_CONFLICTS.map((item, i) => (
                  <ConflictRow key={item.clause} item={item} delay={600 + i * 120} />
                ))}
              </div>

              {/* Scan line animation */}
              <div className="mt-4 h-px bg-gradient-to-r from-transparent via-amber-500/40 to-transparent animate-scan" />

              {/* Bottom label */}
              <p className="text-[10px] font-mono text-zinc-600 mt-3 text-center">
                cross_compare · nli_model · token_attribution
              </p>
            </div>
          </div>

        </div>
      </div>
    </section>
  )
}
