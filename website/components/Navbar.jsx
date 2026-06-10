'use client'
import { useState, useEffect } from 'react'
import { GavelIcon, GithubLogo, ArrowUpRight } from '@phosphor-icons/react'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-zinc-950/90 backdrop-blur-md border-b border-zinc-800/60'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <a href="#" className="flex items-center gap-2 group">
          <span className="w-7 h-7 rounded-md bg-amber-500/10 border border-amber-500/30 flex items-center justify-center">
            <GavelIcon size={14} weight="fill" className="text-amber-500" />
          </span>
          <span className="text-sm font-semibold tracking-tight text-zinc-100">
            LCR
          </span>
        </a>

        {/* Nav links */}
        <div className="hidden md:flex items-center gap-7">
          {['Features', 'Analyzer', 'Docs'].map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              className="text-xs text-zinc-400 hover:text-zinc-100 transition-colors duration-200 tracking-wide"
            >
              {item}
            </a>
          ))}
        </div>

        {/* CTA */}
        <a
          href="https://github.com/AyushDas4890/Legal-Conflict-Resolver"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 hover:border-zinc-600 text-zinc-200 transition-all duration-200 active:scale-[0.97]"
        >
          <GithubLogo size={13} weight="fill" />
          GitHub
          <ArrowUpRight size={11} className="text-zinc-500" />
        </a>
      </div>
    </nav>
  )
}
