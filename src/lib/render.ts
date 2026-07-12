import { Matrix, type Network } from './nn'
import type { Dataset } from './datasets'

/** Pre-build the grid of input coordinates the decision surface is sampled on. */
export function makeGrid(g: number, range: number): Matrix {
  const grid = new Matrix(g * g, 2)
  for (let gy = 0; gy < g; gy++) {
    for (let gx = 0; gx < g; gx++) {
      const x = -range + (gx / (g - 1)) * 2 * range
      const y = range - (gy / (g - 1)) * 2 * range
      const row = (gy * g + gx) * 2
      grid.data[row] = x
      grid.data[row + 1] = y
    }
  }
  return grid
}

const INK = [10, 15, 30]
const C0 = [34, 211, 238] // cyan  → class 0
const C1 = [34, 197, 94] // green → class 1

/**
 * Paint the network's decision surface: sample it on a coarse grid, blend a
 * colour per cell by how confidently it favours a class, upscale it smoothly,
 * then dot the training data on top.
 */
export function paintBoundary(
  ctx: CanvasRenderingContext2D,
  off: HTMLCanvasElement,
  net: Network,
  grid: Matrix,
  g: number,
  data: Dataset,
  size: number,
  range: number,
): void {
  const probs = net.predict(grid)
  const offCtx = off.getContext('2d')!
  const img = offCtx.createImageData(g, g)
  for (let i = 0; i < g * g; i++) {
    const p1 = probs.get(i, 1)
    const conf = Math.abs(2 * p1 - 1) * 0.6
    const c = p1 > 0.5 ? C1 : C0
    const o = i * 4
    img.data[o] = INK[0] + (c[0] - INK[0]) * conf
    img.data[o + 1] = INK[1] + (c[1] - INK[1]) * conf
    img.data[o + 2] = INK[2] + (c[2] - INK[2]) * conf
    img.data[o + 3] = 255
  }
  offCtx.putImageData(img, 0, 0)

  ctx.imageSmoothingEnabled = true
  ctx.clearRect(0, 0, size, size)
  ctx.drawImage(off, 0, 0, g, g, 0, 0, size, size)

  // axes
  ctx.strokeStyle = 'rgba(148,163,184,0.18)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(size / 2, 0)
  ctx.lineTo(size / 2, size)
  ctx.moveTo(0, size / 2)
  ctx.lineTo(size, size / 2)
  ctx.stroke()

  // data points
  const toPx = (x: number) => ((x + range) / (2 * range)) * size
  const toPy = (y: number) => ((range - y) / (2 * range)) * size
  for (let i = 0; i < data.X.length; i++) {
    const px = toPx(data.X[i][0])
    const py = toPy(data.X[i][1])
    ctx.beginPath()
    ctx.arc(px, py, 3.1, 0, Math.PI * 2)
    ctx.fillStyle = data.y[i] === 1 ? '#22c55e' : '#22d3ee'
    ctx.fill()
    ctx.lineWidth = 1
    ctx.strokeStyle = 'rgba(10,15,30,0.85)'
    ctx.stroke()
  }
}
