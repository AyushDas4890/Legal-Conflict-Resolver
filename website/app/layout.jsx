import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import './globals.css'

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
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="bg-zinc-950 text-zinc-50 font-sans antialiased">
        {children}
      </body>
    </html>
  )
}
