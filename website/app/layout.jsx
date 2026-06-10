import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata = {
  title: 'Legal Conflict Resolver — NLP-Powered Contract Analysis',
  description:
    'Detect contradictions between legal document clauses using DeBERTa-v3 NLI. Upload two contracts, get AI-powered conflict analysis in seconds.',
  keywords: ['legal AI', 'contract analysis', 'NLP', 'conflict detection', 'legal tech'],
  openGraph: {
    title: 'Legal Conflict Resolver',
    description: 'AI-powered contract clause conflict detection',
    type: 'website',
  },
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="bg-zinc-950 text-zinc-50 font-sans antialiased">
        {children}
      </body>
    </html>
  )
}
