import Navbar from '../components/Navbar'
import Hero from '../components/Hero'
import Features from '../components/Features'
import Analyzer from '../components/Analyzer'
import Footer from '../components/Footer'

export default function Home() {
  return (
    <main className="min-h-screen bg-zinc-950">
      <Navbar />
      <Hero />
      <Features />
      <Analyzer />
      <Footer />
    </main>
  )
}
