'use client'
import { useState, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'

const EASE = [0.16, 1, 0.3, 1]
import {
  UploadSimple,
  Warning,
  CheckCircle,
  X,
  CircleNotch,
  ArrowRight,
  FileText,
  Lightbulb,
} from '@phosphor-icons/react'

const MOCK_RESULT = {
  document_a: 'contract_a.pdf',
  document_b: 'contract_b.pdf',
  total_pairs_compared: 148,
  conflicts_found: 3,
  conflicts: [
    {
      clause_a_text:
        'In no event shall either party be liable for indirect, incidental, or consequential damages arising out of this agreement.',
      clause_b_text:
        'Each party shall be fully liable for all damages, including indirect and consequential damages, arising from any breach of this agreement.',
      predicted_label: 'contradiction',
      contradiction_prob: 0.9412,
      severity: 3,
      severity_label: 'HIGH',
      clause_type_a: 'liability',
      clause_type_b: 'liability',
      top_conflict_tokens_a: ['no event', 'indirect', 'consequential damages'],
      top_conflict_tokens_b: ['fully liable', 'all damages', 'indirect and consequential'],
      recommendation:
        'Reconcile liability caps. Document B creates unlimited liability exposure that directly contradicts the cap in Document A. Recommend aligning on a mutual exclusion with agreed carve-outs.',
    },
    {
      clause_a_text:
        'Either party may terminate this agreement upon 30 days written notice.',
      clause_b_text:
        'This agreement shall remain in force for a minimum of 24 months and may not be terminated without cause during this period.',
      predicted_label: 'contradiction',
      contradiction_prob: 0.8741,
      severity: 2,
      severity_label: 'MEDIUM',
      clause_type_a: 'termination',
      clause_type_b: 'termination',
      top_conflict_tokens_a: ['30 days', 'written notice'],
      top_conflict_tokens_b: ['minimum of 24 months', 'may not be terminated'],
      recommendation:
        'Document A allows at-will termination with notice; Document B imposes a lock-in period. Define whether the lock-in or the termination right governs, and specify any carve-outs for material breach.',
    },
    {
      clause_a_text:
        'All intellectual property created under this agreement shall belong solely to the Client.',
      clause_b_text:
        'The Service Provider retains ownership of all deliverables and grants the Client a non-exclusive license to use them.',
      predicted_label: 'contradiction',
      contradiction_prob: 0.9183,
      severity: 4,
      severity_label: 'CRITICAL',
      clause_type_a: 'intellectual_property',
      clause_type_b: 'intellectual_property',
      top_conflict_tokens_a: ['belong solely to the Client'],
      top_conflict_tokens_b: ['Service Provider retains ownership', 'non-exclusive license'],
      recommendation:
        'Direct ownership conflict. One party assumes assignment; the other assumes license. This must be resolved before signing — ambiguity creates litigation risk over core deliverables.',
    },
  ],
}

const SEVERITY_STYLES = {
  CRITICAL: {
    badge: 'text-red-400 bg-red-400/10 border-red-400/25',
    dot: 'bg-red-400',
    bar: 'bg-red-400',
  },
  HIGH: {
    badge: 'text-orange-400 bg-orange-400/10 border-orange-400/25',
    dot: 'bg-orange-400',
    bar: 'bg-orange-400',
  },
  MEDIUM: {
    badge: 'text-amber-400 bg-amber-400/10 border-amber-400/25',
    dot: 'bg-amber-400',
    bar: 'bg-amber-400',
  },
  LOW: {
    badge: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25',
    dot: 'bg-emerald-400',
    bar: 'bg-emerald-400',
  },
}

function HighlightedText({ text, tokens }) {
  if (!tokens || tokens.length === 0) return <span>{text}</span>
  let result = text
  tokens.forEach((token) => {
    result = result.replace(
      new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'),
      `<mark class="token-highlight">$&</mark>`
    )
  })
  return <span dangerouslySetInnerHTML={{ __html: result }} />
}

function DropZone({ label, file, onFile, onClear }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragging(false)
      const f = e.dataTransfer.files[0]
      if (f) onFile(f)
    },
    [onFile]
  )

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !file && inputRef.current?.click()}
      className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed transition-all duration-200 cursor-pointer min-h-[140px] p-5 ${
        file
          ? 'border-amber-500/40 bg-amber-500/5 cursor-default'
          : dragging
          ? 'border-amber-500/60 bg-amber-500/10'
          : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/40 hover:bg-zinc-900/70'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt"
        className="sr-only"
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
      {file ? (
        <div className="flex items-center gap-3 w-full">
          <div className="w-9 h-9 rounded-md bg-amber-500/10 border border-amber-500/30 flex items-center justify-center shrink-0">
            <FileText size={16} weight="duotone" className="text-amber-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-zinc-200 truncate">{file.name}</p>
            <p className="text-xs text-zinc-500 mt-0.5">
              {(file.size / 1024).toFixed(1)} KB
            </p>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClear() }}
            className="p-1.5 rounded-md hover:bg-zinc-700 text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <X size={12} weight="bold" />
          </button>
        </div>
      ) : (
        <>
          <UploadSimple size={22} className="text-zinc-600 mb-2" />
          <p className="text-xs font-medium text-zinc-400">{label}</p>
          <p className="text-[11px] text-zinc-600 mt-1">PDF · DOCX · TXT</p>
        </>
      )}
    </div>
  )
}

function ConflictCard({ conflict, index }) {
  const [open, setOpen] = useState(false)
  const styles = SEVERITY_STYLES[conflict.severity_label] || SEVERITY_STYLES.MEDIUM

  return (
    <div className="border border-zinc-800 rounded-lg overflow-hidden bg-zinc-900/40">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-zinc-800/40 transition-colors"
      >
        <div className="flex items-center gap-3 min-w-0">
          <span className={`w-2 h-2 rounded-full shrink-0 ${styles.dot}`} />
          <span className="text-xs font-mono text-zinc-500 shrink-0">
            #{String(index + 1).padStart(2, '0')}
          </span>
          <span className="text-sm text-zinc-300 font-medium truncate">
            {conflict.clause_type_a.replace('_', ' ')} · {conflict.clause_type_b.replace('_', ' ')}
          </span>
        </div>
        <div className="flex items-center gap-3 shrink-0 ml-3">
          <span className="text-xs font-mono text-zinc-500">
            {(conflict.contradiction_prob * 100).toFixed(1)}%
          </span>
          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${styles.badge}`}>
            {conflict.severity_label}
          </span>
          <ArrowRight
            size={12}
            className={`text-zinc-600 transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
          />
        </div>
      </button>

      {open && (
        <div className="border-t border-zinc-800 p-4 space-y-4">
          {/* Probability bar */}
          <div>
            <div className="flex justify-between mb-1.5">
              <span className="text-[10px] font-mono text-zinc-600">contradiction probability</span>
              <span className="text-[10px] font-mono text-zinc-400">
                {(conflict.contradiction_prob * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ${styles.bar}`}
                style={{ width: `${conflict.contradiction_prob * 100}%` }}
              />
            </div>
          </div>

          {/* Clause comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[
              { label: 'Document A', text: conflict.clause_a_text, tokens: conflict.top_conflict_tokens_a },
              { label: 'Document B', text: conflict.clause_b_text, tokens: conflict.top_conflict_tokens_b },
            ].map(({ label, text, tokens }) => (
              <div key={label} className="bg-zinc-950/60 rounded-md p-3 border border-zinc-800/60">
                <p className="text-[10px] font-mono text-zinc-600 mb-2 uppercase">{label}</p>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  <HighlightedText text={text} tokens={tokens} />
                </p>
              </div>
            ))}
          </div>

          {/* Recommendation */}
          <div className="flex gap-2.5 bg-amber-500/6 border border-amber-500/15 rounded-md p-3">
            <Lightbulb size={14} weight="fill" className="text-amber-400 shrink-0 mt-0.5" />
            <p className="text-xs text-zinc-300 leading-relaxed">{conflict.recommendation}</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default function Analyzer() {
  const [fileA, setFileA] = useState(null)
  const [fileB, setFileB] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const canAnalyze = fileA && fileB && !loading

  const handleAnalyze = async () => {
    if (!canAnalyze) return
    setLoading(true)
    setError(null)
    setResult(null)

    const apiUrl = process.env.NEXT_PUBLIC_API_URL

    if (!apiUrl) {
      // Demo mode — use mock data with artificial delay
      await new Promise((r) => setTimeout(r, 1800))
      setResult(MOCK_RESULT)
      setLoading(false)
      return
    }

    try {
      const form = new FormData()
      form.append('doc_a', fileA)
      form.append('doc_b', fileB)

      const res = await fetch(`${apiUrl}/analyze`, { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section id="analyzer" className="py-28 px-6 border-t border-zinc-800/60">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.6, ease: EASE }}
          className="mb-14 max-w-lg"
        >
          <h2 className="text-3xl md:text-4xl font-bold tracking-tighter text-zinc-50 leading-tight">
            Upload. Compare.<br />Resolve.
          </h2>
          <p className="text-zinc-400 text-sm leading-relaxed mt-4">
            Drop two contracts below. Results include severity scoring, clause type labels, and token-level conflict attribution.
          </p>
          {!process.env.NEXT_PUBLIC_API_URL && (
            <p className="text-[11px] font-mono text-amber-500/60 mt-3 bg-amber-500/6 border border-amber-500/15 rounded px-3 py-1.5">
              Demo mode — results are mock data. Set NEXT_PUBLIC_API_URL to connect the live backend.
            </p>
          )}
        </motion.div>

        {/* Upload area */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <DropZone
            label="Document A — drop here"
            file={fileA}
            onFile={setFileA}
            onClear={() => setFileA(null)}
          />
          <DropZone
            label="Document B — drop here"
            file={fileB}
            onFile={setFileB}
            onClear={() => setFileB(null)}
          />
        </div>

        {/* Analyze button */}
        <button
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          className={`flex items-center gap-2 px-6 py-3 rounded-md text-sm font-semibold transition-all duration-200 ${
            canAnalyze
              ? 'bg-amber-500 hover:bg-amber-400 text-zinc-950 active:scale-[0.97] active:-translate-y-[1px]'
              : 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
          }`}
        >
          {loading ? (
            <>
              <CircleNotch size={15} className="animate-spin" />
              Analyzing…
            </>
          ) : (
            <>
              Analyze conflicts
              <ArrowRight size={14} weight="bold" />
            </>
          )}
        </button>

        {/* Error state */}
        {error && (
          <div className="mt-6 flex items-start gap-2.5 bg-red-500/8 border border-red-500/20 rounded-lg p-4">
            <Warning size={15} weight="fill" className="text-red-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-300">Analysis failed</p>
              <p className="text-xs text-zinc-500 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-10 space-y-4">
            {/* Summary bar */}
            <div className="flex flex-wrap items-center gap-6 pb-6 border-b border-zinc-800/60">
              <div className="flex items-center gap-2">
                <CheckCircle size={15} weight="fill" className="text-emerald-500" />
                <span className="text-sm text-zinc-300">
                  <span className="font-mono font-semibold text-zinc-100">{result.total_pairs_compared}</span>
                  {' '}pairs compared
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Warning size={15} weight="fill" className="text-amber-500" />
                <span className="text-sm text-zinc-300">
                  <span className="font-mono font-semibold text-zinc-100">{result.conflicts_found}</span>
                  {' '}conflicts found
                </span>
              </div>
            </div>

            {/* Conflict cards */}
            <div className="space-y-3">
              {result.conflicts.map((c, i) => (
                <ConflictCard key={i} conflict={c} index={i} />
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
