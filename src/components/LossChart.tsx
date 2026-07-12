type Props = { history: { loss: number; acc: number }[] }

const W = 320
const H = 90

/** Tiny SVG sparkline of loss (green) and accuracy (cyan) over training steps. */
export default function LossChart({ history }: Props) {
  if (history.length < 2) {
    return (
      <div className="flex h-[90px] items-center justify-center rounded-lg border border-line bg-ink/60 text-xs text-fog">
        loss curve appears once training starts
      </div>
    )
  }

  const maxLoss = Math.max(...history.map((h) => h.loss), 0.01)
  const n = history.length
  const x = (i: number) => (i / (n - 1)) * W

  const lossPath = history
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${(H - (h.loss / maxLoss) * (H - 6) - 3).toFixed(1)}`)
    .join(' ')
  const accPath = history
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${(H - h.acc * (H - 6) - 3).toFixed(1)}`)
    .join(' ')

  return (
    <div className="rounded-lg border border-line bg-ink/60 p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Loss and accuracy over time">
        <path d={accPath} fill="none" stroke="#22d3ee" strokeWidth="1.5" opacity="0.85" />
        <path d={lossPath} fill="none" stroke="#22c55e" strokeWidth="1.5" />
      </svg>
      <div className="flex justify-between px-1 font-mono text-[11px] text-fog">
        <span className="text-live">— loss</span>
        <span className="text-cyan">— accuracy</span>
      </div>
    </div>
  )
}
