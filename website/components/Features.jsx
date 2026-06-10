import {
  FileText,
  Scissors,
  Brain,
  ChartBar,
  Sparkle,
  Scales,
} from '@phosphor-icons/react'

const FEATURES = [
  {
    icon: FileText,
    title: 'Multi-format ingestion',
    body: 'Extracts raw text from PDF and DOCX files server-side. No client-side parsing, no quality loss from OCR quirks.',
    span: 'col-span-1',
  },
  {
    icon: Scissors,
    title: 'Clause segmentation',
    body: 'Splits documents by sentence boundaries, producing clean clause units ready for pairwise comparison.',
    span: 'col-span-1',
  },
  {
    icon: Brain,
    title: 'DeBERTa-v3 NLI',
    body: 'Microsoft\'s DeBERTa-v3-large fine-tuned on MNLI + SNLI. Three-way classification: entailment, neutral, or contradiction — per clause pair.',
    span: 'col-span-1 md:col-span-2',
  },
  {
    icon: ChartBar,
    title: 'Severity scoring',
    body: 'Contradiction probability drives a four-band severity score (CRITICAL · HIGH · MEDIUM · LOW) so legal teams triage fast.',
    span: 'col-span-1',
  },
  {
    icon: Sparkle,
    title: 'Token attribution',
    body: 'Integrated gradients surface the exact spans driving the prediction — no black box. You see why each conflict was flagged.',
    span: 'col-span-1',
  },
  {
    icon: Scales,
    title: 'Clause type tagging',
    body: 'Each clause is automatically labelled: payment, liability, IP, termination, governing law, and 11 more — for structured downstream review.',
    span: 'col-span-1',
  },
]

export default function Features() {
  return (
    <section id="features" className="py-28 px-6 border-t border-zinc-800/60">
      <div className="max-w-7xl mx-auto">

        {/* Section header — left-aligned */}
        <div className="mb-14 max-w-lg">
          <p className="text-xs font-mono text-amber-500/70 uppercase tracking-widest mb-3">
            Pipeline
          </p>
          <h2 className="text-3xl md:text-4xl font-bold tracking-tighter text-zinc-50 leading-tight">
            Five-phase analysis,<br />end-to-end.
          </h2>
          <p className="text-zinc-400 text-sm leading-relaxed mt-4">
            From raw file to token-level conflict evidence — each stage runs deterministically with no hallucination risk.
          </p>
        </div>

        {/* Asymmetric grid — DESIGN_VARIANCE=8 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-zinc-800/40 rounded-xl overflow-hidden">
          {FEATURES.map(({ icon: Icon, title, body, span }) => (
            <div
              key={title}
              className={`${span} bg-zinc-950 p-7 hover:bg-zinc-900/60 transition-colors duration-300 group`}
            >
              <div className="w-9 h-9 rounded-lg bg-zinc-800/80 border border-zinc-700/50 flex items-center justify-center mb-4 group-hover:border-amber-500/30 group-hover:bg-amber-500/8 transition-all duration-300">
                <Icon size={16} weight="duotone" className="text-zinc-400 group-hover:text-amber-400 transition-colors duration-300" />
              </div>
              <h3 className="text-sm font-semibold text-zinc-100 tracking-tight mb-2">{title}</h3>
              <p className="text-sm text-zinc-500 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>

      </div>
    </section>
  )
}
