import { useEffect, useMemo, useRef, useState } from 'react'
import { Eraser } from 'lucide-react'
import { Matrix, Network, type SerializedNet } from '../lib/nn'
import weightsJson from '../weights/digits.json'

const weights = weightsJson as unknown as SerializedNet & { testAccuracy?: number }

const SIZE = 280 // drawing canvas (px)

/** Downsample the 280×280 drawing to 28×28 intensities, then shift the digit's
 *  centre of mass to the middle — the same normalisation MNIST digits get. */
function preprocess(main: HTMLCanvasElement): { input: number[]; hasInk: boolean } {
  const off = document.createElement('canvas')
  off.width = 28
  off.height = 28
  const octx = off.getContext('2d')!
  octx.drawImage(main, 0, 0, 28, 28)
  const px = octx.getImageData(0, 0, 28, 28).data

  const grid = new Float64Array(28 * 28)
  let total = 0
  let cx = 0
  let cy = 0
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = px[(y * 28 + x) * 4] / 255 // white ink on black → intensity
      grid[y * 28 + x] = v
      total += v
      cx += x * v
      cy += y * v
    }
  }
  if (total < 1) return { input: Array.from(grid), hasInk: false }

  const shiftX = Math.round(14 - cx / total)
  const shiftY = Math.round(14 - cy / total)
  const centred = new Float64Array(28 * 28)
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sx = x - shiftX
      const sy = y - shiftY
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) centred[y * 28 + x] = grid[sy * 28 + sx]
    }
  }
  return { input: Array.from(centred), hasInk: true }
}

export default function DigitRecognizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawing = useRef(false)
  const net = useMemo(() => Network.deserialize(weights), [])
  const [probs, setProbs] = useState<number[] | null>(null)
  const testAcc = weights.testAccuracy ?? 0

  const clear = () => {
    const ctx = canvasRef.current!.getContext('2d')!
    ctx.fillStyle = '#0a0f1e'
    ctx.fillRect(0, 0, SIZE, SIZE)
    setProbs(null)
  }

  useEffect(() => {
    clear()
  }, [])

  const predict = () => {
    const { input, hasInk } = preprocess(canvasRef.current!)
    if (!hasInk) {
      setProbs(null)
      return
    }
    const out = net.predict(new Matrix(1, 784, Float64Array.from(input)))
    setProbs(Array.from({ length: 10 }, (_, i) => out.get(0, i)))
  }

  const pos = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / rect.width) * SIZE,
      y: ((e.clientY - rect.top) / rect.height) * SIZE,
    }
  }

  const start = (e: React.PointerEvent) => {
    drawing.current = true
    canvasRef.current!.setPointerCapture(e.pointerId)
    const ctx = canvasRef.current!.getContext('2d')!
    const { x, y } = pos(e)
    ctx.beginPath()
    ctx.moveTo(x, y)
  }
  const move = (e: React.PointerEvent) => {
    if (!drawing.current) return
    const ctx = canvasRef.current!.getContext('2d')!
    const { x, y } = pos(e)
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 20
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineTo(x, y)
    ctx.stroke()
  }
  const end = () => {
    if (!drawing.current) return
    drawing.current = false
    predict()
  }

  const top = probs ? probs.indexOf(Math.max(...probs)) : null

  return (
    <div className="grid gap-6 sm:grid-cols-[280px_1fr]">
      <div className="space-y-3">
        <canvas
          ref={canvasRef}
          width={SIZE}
          height={SIZE}
          onPointerDown={start}
          onPointerMove={move}
          onPointerUp={end}
          onPointerLeave={end}
          className="aspect-square w-full max-w-[280px] touch-none rounded-2xl border border-line bg-ink"
          aria-label="Draw a digit here"
        />
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-fog">draw a digit 0–9</span>
          <button
            onClick={clear}
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-line px-3.5 py-1.5 font-mono text-xs text-fog transition-colors hover:border-fog/40 hover:text-snow"
          >
            <Eraser size={13} /> clear
          </button>
        </div>
      </div>

      <div className="flex flex-col">
        <div className="mb-4 flex items-baseline gap-3">
          <span className="font-mono text-sm text-fog">prediction</span>
          <span className="font-mono text-6xl font-bold leading-none text-live">
            {top ?? '–'}
          </span>
        </div>

        <div className="flex-1 space-y-1.5">
          {Array.from({ length: 10 }, (_, d) => {
            const p = probs ? probs[d] : 0
            return (
              <div key={d} className="flex items-center gap-3">
                <span className={`w-4 text-right font-mono text-sm ${d === top ? 'text-live' : 'text-fog'}`}>{d}</span>
                <div className="h-3 flex-1 overflow-hidden rounded-full bg-ink">
                  <div
                    className={`h-full rounded-full transition-all duration-200 ${d === top ? 'bg-live' : 'bg-slate-600'}`}
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-xs tabular-nums text-fog">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            )
          })}
        </div>

        <p className="mt-4 rounded-lg border border-line bg-ink/60 p-3 text-[13px] leading-relaxed text-fog">
          This 784→64→32→10 network was trained on MNIST to{' '}
          <span className="text-snow">{(testAcc * 100).toFixed(1)}% test accuracy</span> using the
          same from-scratch engine — offline, then its weights were saved and loaded here. Inference
          runs entirely in your browser.
        </p>
      </div>
    </div>
  )
}
