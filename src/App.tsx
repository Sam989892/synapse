import { motion } from 'framer-motion'
import { Code2, Cpu, PenLine, Boxes } from 'lucide-react'
import Playground from './components/Playground'
import DigitRecognizer from './components/DigitRecognizer'

const layers = [
  { file: 'matrix.ts', desc: 'Float64Array-backed matrix: matmul, transpose, broadcast, Gaussian init' },
  { file: 'layer.ts', desc: 'Dense forward/backward + ReLU · tanh · sigmoid with their gradients' },
  { file: 'losses.ts', desc: 'Numerically-stable softmax + fused cross-entropy gradient' },
  { file: 'optimizer.ts', desc: 'SGD with momentum, and Adam (bias-corrected moments)' },
  { file: 'network.ts', desc: 'Sequential model: forward, backprop, train step, save/load' },
]

export default function App() {
  return (
    <div className="min-h-screen">
      {/* header */}
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <span className="flex items-center gap-2.5">
          <Logo />
          <span className="font-mono text-lg font-bold tracking-tight">synapse</span>
        </span>
        <a
          href="#"
          className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 font-mono text-sm text-fog transition-colors hover:border-fog/40 hover:text-snow"
        >
          <Code2 size={16} /> source
        </a>
      </header>

      {/* hero */}
      <section className="mx-auto max-w-6xl px-6 pt-10 pb-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-ink-2/60 px-4 py-1.5 font-mono text-xs text-live">
            <Cpu size={13} /> no tensorflow · no pytorch · no api
          </span>
          <h1 className="mx-auto mt-6 max-w-3xl font-mono text-4xl font-bold leading-tight tracking-tight text-balance sm:text-5xl">
            A neural network,
            <span className="bg-gradient-to-r from-live to-cyan bg-clip-text text-transparent"> from scratch.</span>
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-lg leading-relaxed text-fog">
            Every matrix multiply, every gradient, the whole backprop pass — hand-written in
            TypeScript. Watch it learn below, then draw a digit and let it read your handwriting.
          </p>
        </motion.div>
      </section>

      {/* playground */}
      <Showcase
        icon={Boxes}
        eyebrow="01 — the training playground"
        title="Watch backpropagation learn"
        sub="Pick a dataset, shape the network, and hit Train. The decision boundary is the model thinking in real time."
      >
        <Playground />
      </Showcase>

      {/* digit recognizer */}
      <Showcase
        icon={PenLine}
        eyebrow="02 — it reads handwriting"
        title="Draw a digit"
        sub="The same engine, trained on MNIST offline, now runs live in your browser. No round-trip to a server."
      >
        <DigitRecognizer />
      </Showcase>

      {/* engine */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="mb-8">
          <span className="font-mono text-xs uppercase tracking-wide text-live">03 — under the hood</span>
          <h2 className="mt-2 font-mono text-2xl font-bold sm:text-3xl">The whole engine is ~500 lines</h2>
          <p className="mt-2 max-w-2xl text-fog">
            No dependencies in the maths — not even a linear-algebra library. This is what PyTorch
            does, in miniature, readable end to end.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {layers.map((l) => (
            <div key={l.file} className="rounded-xl border border-line bg-ink-2/50 p-4">
              <div className="font-mono text-sm font-semibold text-live-2">{l.file}</div>
              <p className="mt-1.5 text-[14px] leading-relaxed text-fog">{l.desc}</p>
            </div>
          ))}
          <div className="flex items-center rounded-xl border border-live/30 bg-live/[0.06] p-4">
            <p className="text-[14px] leading-relaxed text-snow">
              Verified: trains spiral / XOR / circles / moons to <span className="text-live">99–100%</span>,
              and MNIST digits to <span className="text-live">97.6%</span> test accuracy.
            </p>
          </div>
        </div>
      </section>

      <footer className="border-t border-line">
        <div className="mx-auto max-w-6xl px-6 py-8 text-center font-mono text-xs text-fog">
          Concept project — built from scratch by Saiyed (Sam) Madni · React · TypeScript · zero ML libraries
        </div>
      </footer>
    </div>
  )
}

function Showcase({
  icon: Icon,
  eyebrow,
  title,
  sub,
  children,
}: {
  icon: typeof Boxes
  eyebrow: string
  title: string
  sub: string
  children: React.ReactNode
}) {
  return (
    <section className="border-t border-line bg-ink-2/20">
      <div className="mx-auto max-w-6xl px-6 py-14">
        <div className="mb-8 flex items-start gap-3">
          <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-line bg-ink-3 text-live">
            <Icon size={18} />
          </span>
          <div>
            <span className="font-mono text-xs uppercase tracking-wide text-live">{eyebrow}</span>
            <h2 className="mt-1 font-mono text-2xl font-bold sm:text-3xl">{title}</h2>
            <p className="mt-1.5 max-w-2xl text-fog">{sub}</p>
          </div>
        </div>
        {children}
      </div>
    </section>
  )
}

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" className="fill-ink-3" />
      <circle cx="8" cy="9" r="2.4" fill="#22c55e" />
      <circle cx="8" cy="23" r="2.4" fill="#22c55e" />
      <circle cx="16" cy="16" r="2.4" fill="#4ade80" />
      <circle cx="24" cy="16" r="2.4" fill="#22d3ee" />
      <g stroke="#334155" strokeWidth="1.1">
        <line x1="8" y1="9" x2="16" y2="16" />
        <line x1="8" y1="23" x2="16" y2="16" />
        <line x1="16" y1="16" x2="24" y2="16" />
      </g>
    </svg>
  )
}
