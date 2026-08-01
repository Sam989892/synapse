/**
 * Reader for the canonical MNIST IDX files (the real dataset — 60,000 training
 * and 10,000 test images, as published by LeCun/Cortes/Burges). This is the
 * original binary format, not an npm-bundled subset:
 *
 *   images: [magic=2051][count][rows=28][cols=28][pixel bytes...]   (uint8, big-endian header)
 *   labels: [magic=2049][count][label bytes...]
 *
 * Pixels are normalised to [0, 1] (byte / 255). MNIST digits are white ink on a
 * black background and are already center-of-mass centred in the 28×28 frame —
 * exactly the representation the browser produces from a drawn digit, so weights
 * trained here drop straight into inference.
 *
 * Fetch the files first with scripts/fetch-mnist.sh.
 */
import { readFileSync } from 'node:fs'
import { Matrix } from '../src/lib/nn'

/** Load an IDX image file into a (count × 784) Matrix, pixels scaled to [0,1]. */
export function loadImages(path: string): Matrix {
  const buf = readFileSync(path)
  const magic = buf.readUInt32BE(0)
  if (magic !== 2051) throw new Error(`${path}: bad image magic ${magic} (expected 2051)`)
  const count = buf.readUInt32BE(4)
  const rows = buf.readUInt32BE(8)
  const cols = buf.readUInt32BE(12)
  const size = rows * cols
  const data = new Float64Array(count * size)
  let off = 16
  for (let i = 0; i < count * size; i++) data[i] = buf[off + i] / 255
  return new Matrix(count, size, data)
}

/** Load an IDX label file into a plain number[] of class ids 0–9. */
export function loadLabels(path: string): number[] {
  const buf = readFileSync(path)
  const magic = buf.readUInt32BE(0)
  if (magic !== 2049) throw new Error(`${path}: bad label magic ${magic} (expected 2049)`)
  const count = buf.readUInt32BE(4)
  const out = new Array<number>(count)
  for (let i = 0; i < count; i++) out[i] = buf[8 + i]
  return out
}
