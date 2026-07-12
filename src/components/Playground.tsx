import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Play, Pause, RotateCcw, StepForward, Plus, Minus } from 'lucide-react'
import { Matrix, Network, Adam, SGD, type Activation, type Optimizer } from '../lib/nn'
import { makeDataset, type DatasetId, RANGE } from '../lib/datasets'
import { makeGrid, paintBoundary } from '../lib/render'
import LossChart from './LossChart'

const SIZE = 460
const GRID = 64
const DATASETS: { id: DatasetId; label: string }[] = [
  { id: 'spiral', label: 'Spiral' },
  { id: 'circles', label: 'Circles' },
  { id: 'xor', label: 'XOR' },
  { id: 'moons', label: 'Moons' },
  { id: 'gaussian', label: 'Gaussian' },
]

export default function Playground() {
  const [dataset, setDataset] = useState<DatasetId>('spiral')
  const [noise, setNoise] = useState(0.15)
  const [hidden, setHidden] = useState<number[]>([8, 8])
  const [activation, setActivation] = useState<Activation>('tanh')
  const [optName, setOptName] = useState<'adam' | 'sgd'>('adam')
  const [lr, setLr] = useState(0.03)
  const [running, setRunning] = useState(false)

  const [step, setStep] = useState(0)
  const [loss, setLoss] = useState(0)
  const [acc, setAcc] = useState(0)
  const [history, setHistory] = useState<{ loss: number; acc: number }[]>([])

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const offRef = useRef<HTMLCanvasElement | null>(null)
  const netRef = useRef<Network | null>(null)
  const optRef = useRef<Optimizer | null>(null)
  const dataRef = useRef(makeDataset(dataset, 240, noise))
  const XRef = useRef<Matrix>(Matrix.from(dataRef.current.X))
  const grid = useMemo(() => makeGrid(GRID, RANGE), [])
  const runningRef = useRef(false)

  if (!offRef.current && typeof document !== 'undefined') {
    const c = document.createElement('canvas')
    c.width = GRID
    c.height = GRID
    offRef.current = c
  }

  const draw = useCallback(() => {
    const ctx = canvasRef.current?.getContext('2d')
    if (ctx && netRef.current && offRef.current)
      paintBoundary(ctx, offRef.current, netRef.current, grid, GRID, dataRef.current, SIZE, RANGE)
  }, [grid])

  const rebuild = useCallback(() => {
    const data = makeDataset(dataset, 240, noise)
    dataRef.current = data
    XRef.current = Matrix.from(data.X)
    netRef.current = Network.mlp([2, ...hidden, 2], activation)
    optRef.current = optName === 'adam' ? new Adam(lr) : new SGD(lr)
    setStep(0)
    setLoss(0)
    setAcc(0)
    setHistory([])
    draw()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, noise, hidden, activation, optName, draw])

  // rebuild whenever the architecture / data config changes
  useEffect(() => {
    rebuild()
  }, [rebuild])

  // learning rate can change live without a rebuild
  useEffect(() => {
    if (optRef.current) optRef.current.lr = lr
  }, [lr])

  const doSteps = useCallback((count: number) => {
    const net = netRef.current
    const opt = optRef.current
    if (!net || !opt) return { loss: 0, acc: 0 }
    let last = { loss: 0, acc: 0 }
    for (let i = 0; i < count; i++) last = net.trainStep(XRef.current, dataRef.current.y, opt)
    return last
  }, [])

  // training loop
  useEffect(() => {
    runningRef.current = running
    if (!running) return
    let raf = 0
    let frame = 0
    const tick = () => {
      const r = doSteps(3)
      draw()
      frame++
      if (frame % 3 === 0) {
        setStep((s) => s + 3 * 3)
        setLoss(r.loss)
        setAcc(r.acc)
        setHistory((h) => [...h.slice(-160), { loss: r.loss, acc: r.acc }])
      }
      if (runningRef.current) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [running, doSteps, draw])

  const singleStep = () => {
    const r = doSteps(1)
    setStep((s) => s + 1)
    setLoss(r.loss)
    setAcc(r.acc)
    setHistory((h) => [...h.slice(-160), r])
    draw()
  }

  const addNeuron = (i: number) =>
    setHidden((h) => h.map((n, k) => (k === i ? Math.min(n + 1, 12) : n)))
  const removeNeuron = (i: number) =>
    setHidden((h) => h.map((n, k) => (k === i ? Math.max(n - 1, 1) : n)))
  const addLayer = () => setHidden((h) => (h.length < 4 ? [...h, 6] : h))
  const removeLayer = () => setHidden((h) => (h.length > 1 ? h.slice(0, -1) : h))

  return (
    <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
      {/* controls */}
      <div className="space-y-5 rounded-2xl border border-line bg-ink-2/60 p-5">
        <Field label="Dataset">
          <div className="grid grid-cols-2 gap-1.5">
            {DATASETS.map((d) => (
              <button
                key={d.id}
                onClick={() => setDataset(d.id)}
                className={chip(dataset === d.id)}
              >
                {d.label}
              </button>
            ))}
          </div>
        </Field>

        <Field label={`Noise · ${noise.toFixed(2)}`}>
          <input type="range" min={0} max={0.6} step={0.01} value={noise} onChange={(e) => setNoise(+e.target.value)} className="w-full accent-live" />
        </Field>

        <Field label="Hidden layers">
          <div className="space-y-2">
            {hidden.map((n, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-14 font-mono text-xs text-fog">L{i + 1}</span>
                <button onClick={() => removeNeuron(i)} className={stepBtn}><Minus size={13} /></button>
                <span className="w-8 text-center font-mono text-sm">{n}</span>
                <button onClick={() => addNeuron(i)} className={stepBtn}><Plus size={13} /></button>
                <span className="text-xs text-fog">neurons</span>
              </div>
            ))}
            <div className="flex gap-2 pt-1">
              <button onClick={addLayer} disabled={hidden.length >= 4} className={`${miniBtn} disabled:opacity-40`}>+ layer</button>
              <button onClick={removeLayer} disabled={hidden.length <= 1} className={`${miniBtn} disabled:opacity-40`}>− layer</button>
            </div>
          </div>
        </Field>

        <Field label="Activation">
          <div className="grid grid-cols-3 gap-1.5">
            {(['tanh', 'relu', 'sigmoid'] as Activation[]).map((a) => (
              <button key={a} onClick={() => setActivation(a)} className={chip(activation === a)}>{a}</button>
            ))}
          </div>
        </Field>

        <Field label="Optimizer">
          <div className="grid grid-cols-2 gap-1.5">
            {(['adam', 'sgd'] as const).map((o) => (
              <button key={o} onClick={() => setOptName(o)} className={chip(optName === o)}>{o.toUpperCase()}</button>
            ))}
          </div>
        </Field>

        <Field label={`Learning rate · ${lr.toFixed(3)}`}>
          <input type="range" min={0.001} max={0.2} step={0.001} value={lr} onChange={(e) => setLr(+e.target.value)} className="w-full accent-live" />
        </Field>
      </div>

      {/* canvas + readouts */}
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => setRunning((r) => !r)}
            className="inline-flex cursor-pointer items-center gap-2 rounded-full bg-live px-5 py-2.5 font-mono text-sm font-semibold text-ink transition-colors hover:bg-live-2"
          >
            {running ? <><Pause size={16} /> Pause</> : <><Play size={16} /> Train</>}
          </button>
          <button onClick={singleStep} disabled={running} className={ctrlBtn}><StepForward size={15} /> Step</button>
          <button onClick={rebuild} className={ctrlBtn}><RotateCcw size={15} /> Reset</button>

          <div className="ml-auto flex gap-5 font-mono text-sm">
            <Readout label="step" value={step.toLocaleString()} />
            <Readout label="loss" value={loss.toFixed(3)} tone="text-live" />
            <Readout label="acc" value={`${(acc * 100).toFixed(1)}%`} tone="text-cyan" />
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-[460px_1fr]">
          <canvas
            ref={canvasRef}
            width={SIZE}
            height={SIZE}
            className="aspect-square w-full max-w-[460px] rounded-2xl border border-line bg-ink"
            aria-label="Decision boundary the network has learned"
          />
          <div className="space-y-3">
            <LossChart history={history} />
            <p className="rounded-lg border border-line bg-ink/60 p-3 text-[13px] leading-relaxed text-fog">
              The background is the network's decision surface, sampled on a {GRID}×{GRID} grid and
              coloured by class confidence. Dots are the training data. Every frame runs real
              forward + backprop steps through the hand-written engine — no libraries.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-2 font-mono text-xs uppercase tracking-wide text-fog">{label}</div>
      {children}
    </div>
  )
}

function Readout({ label, value, tone = 'text-snow' }: { label: string; value: string; tone?: string }) {
  return (
    <div className="text-right">
      <div className="text-[11px] uppercase text-fog">{label}</div>
      <div className={`font-semibold tabular-nums ${tone}`}>{value}</div>
    </div>
  )
}

const chip = (active: boolean) =>
  `cursor-pointer rounded-lg border px-2.5 py-1.5 font-mono text-xs capitalize transition-colors ${
    active
      ? 'border-live/60 bg-live/10 text-live-2'
      : 'border-line text-fog hover:border-fog/40 hover:text-snow'
  }`

const stepBtn =
  'flex h-6 w-6 cursor-pointer items-center justify-center rounded border border-line text-fog transition-colors hover:border-fog/40 hover:text-snow'
const miniBtn =
  'cursor-pointer rounded border border-line px-2.5 py-1 font-mono text-xs text-fog transition-colors hover:border-fog/40 hover:text-snow'
const ctrlBtn =
  'inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-line px-4 py-2.5 font-mono text-sm text-snow transition-colors hover:border-fog/40 disabled:cursor-not-allowed disabled:opacity-40'
