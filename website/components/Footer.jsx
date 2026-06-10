import { GavelIcon, GithubLogo } from '@phosphor-icons/react'

export default function Footer() {
  return (
    <footer className="border-t border-zinc-800/60 bg-zinc-950">
      <div className="max-w-7xl mx-auto px-6 py-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
        <div className="flex items-center gap-2">
          <span className="w-6 h-6 rounded bg-amber-500/10 border border-amber-500/30 flex items-center justify-center">
            <GavelIcon size={12} weight="fill" className="text-amber-500" />
          </span>
          <span className="text-sm font-medium text-zinc-300 tracking-tight">
            Legal Conflict Resolver
          </span>
        </div>

        <div className="flex items-center gap-6">
          <span className="text-xs text-zinc-600">
            DeBERTa-v3-large · FastAPI · Next.js
          </span>
          <a
            href="https://github.com/AyushDas4890/Legal-Conflict-Resolver"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-100 transition-colors"
          >
            <GithubLogo size={14} weight="fill" />
            View source
          </a>
        </div>
      </div>
    </footer>
  )
}
