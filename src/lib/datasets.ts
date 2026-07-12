/**
 * 2D toy datasets for the training playground. Each returns points in roughly
 * [-6, 6]² with integer class labels. These are the classic separability tests:
 * a linear model fails on all but "gaussian", so they make the hidden layers
 * visibly earn their keep.
 */
export type Dataset = { X: number[][]; y: number[]; classes: number }
export type DatasetId = 'spiral' | 'circles' | 'xor' | 'gaussian' | 'moons'

const R = 5 // coordinate half-range

function noisy(v: number, noise: number): number {
  return v + (Math.random() - 0.5) * 2 * noise
}

/** Two interleaved spiral arms — the hardest of the set. */
function spiral(n: number, noise: number): Dataset {
  const X: number[][] = []
  const y: number[] = []
  const per = Math.floor(n / 2)
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < per; i++) {
      const r = (i / per) * R
      const t = ((i / per) * 3 * Math.PI + c * Math.PI) + (Math.random() - 0.5) * noise
      X.push([r * Math.sin(t), r * Math.cos(t)])
      y.push(c)
    }
  }
  return { X, y, classes: 2 }
}

/** An inner disc surrounded by an outer ring. */
function circles(n: number, noise: number): Dataset {
  const X: number[][] = []
  const y: number[] = []
  const per = Math.floor(n / 2)
  for (let i = 0; i < per; i++) {
    const t = Math.random() * 2 * Math.PI
    const rIn = Math.random() * 1.8
    X.push([noisy(rIn * Math.cos(t), noise), noisy(rIn * Math.sin(t), noise)])
    y.push(0)
    const rOut = 3.2 + Math.random() * 1.5
    X.push([noisy(rOut * Math.cos(t), noise), noisy(rOut * Math.sin(t), noise)])
    y.push(1)
  }
  return { X, y, classes: 2 }
}

/** The XOR checkerboard — class depends on the sign product of the two axes. */
function xor(n: number, noise: number): Dataset {
  const X: number[][] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const x1 = (Math.random() - 0.5) * 2 * R
    const x2 = (Math.random() - 0.5) * 2 * R
    X.push([noisy(x1, noise), noisy(x2, noise)])
    y.push(x1 * x2 > 0 ? 0 : 1)
  }
  return { X, y, classes: 2 }
}

/** Two well-separated Gaussian blobs — linearly separable baseline. */
function gaussian(n: number, noise: number): Dataset {
  const X: number[][] = []
  const y: number[] = []
  const spread = 1.4 + noise
  for (let i = 0; i < n; i++) {
    const c = i % 2
    const cx = c === 0 ? -2.4 : 2.4
    const cy = c === 0 ? -2.4 : 2.4
    X.push([cx + randn() * spread, cy + randn() * spread])
    y.push(c)
  }
  return { X, y, classes: 2 }
}

/** Two half-moons. */
function moons(n: number, noise: number): Dataset {
  const X: number[][] = []
  const y: number[] = []
  const per = Math.floor(n / 2)
  for (let i = 0; i < per; i++) {
    const t = Math.PI * (i / per)
    X.push([noisy(3.5 * Math.cos(t) - 1.8, noise), noisy(3.5 * Math.sin(t) - 1, noise)])
    y.push(0)
    X.push([noisy(3.5 * Math.cos(t) + 1.8, noise), noisy(-3.5 * Math.sin(t) + 1, noise)])
    y.push(1)
  }
  return { X, y, classes: 2 }
}

const generators: Record<DatasetId, (n: number, noise: number) => Dataset> = {
  spiral,
  circles,
  xor,
  gaussian,
  moons,
}

export function makeDataset(id: DatasetId, n: number, noise: number): Dataset {
  return generators[id](n, noise)
}

export const RANGE = R

function randn(): number {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
