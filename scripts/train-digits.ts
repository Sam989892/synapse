/**
 * Trains the handwritten-digit model OFFLINE using the same from-scratch engine
 * the browser uses for inference. Nothing here is an external service.
 *
 * Data: the real, full MNIST dataset — all 60,000 training images and 10,000
 * held-out test images, read straight from the canonical IDX files. Get them
 * first with:  bash scripts/fetch-mnist.sh
 *
 * Output: src/weights/digits.json (network weights + measured test accuracy).
 * Run: npx tsx scripts/train-digits.ts
 */
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { Matrix, Network, Adam, accuracy } from '../src/lib/nn'
import { loadImages, loadLabels } from './mnist-idx'

const EPOCHS = 15
const BATCH = 64
const LR = 0.001

const here = dirname(fileURLToPath(import.meta.url))
const dataDir = join(here, 'mnist-data')

const trainX = loadImages(join(dataDir, 'train-images-idx3-ubyte')) // 60000 × 784
const trainY = loadLabels(join(dataDir, 'train-labels-idx1-ubyte'))
const testX = loadImages(join(dataDir, 't10k-images-idx3-ubyte')) // 10000 × 784
const testY = loadLabels(join(dataDir, 't10k-labels-idx1-ubyte'))
console.log(`loaded real MNIST: ${trainX.rows} train, ${testX.rows} test (28×28)`)

/** Copy a set of rows out of a big matrix into a fresh (n × cols) batch. */
function gatherRows(M: Matrix, idxs: number[]): Matrix {
  const cols = M.cols
  const out = new Float64Array(idxs.length * cols)
  for (let r = 0; r < idxs.length; r++) {
    const src = idxs[r] * cols
    out.set(M.data.subarray(src, src + cols), r * cols)
  }
  return new Matrix(idxs.length, cols, out)
}

// 784 → 64 → 32 → 10 multilayer perceptron.
const net = Network.mlp([784, 64, 32, 10], 'relu')
const opt = new Adam(LR)

const order = [...Array(trainX.rows).keys()]
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  // shuffle indices each epoch (Fisher–Yates)
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  let loss = 0
  let batches = 0
  for (let b = 0; b < order.length; b += BATCH) {
    const slice = order.slice(b, b + BATCH)
    const xb = gatherRows(trainX, slice)
    const yb = slice.map((k) => trainY[k])
    loss += net.trainStep(xb, yb, opt).loss
    batches++
  }
  const testAcc = accuracy(net.predict(testX), testY)
  console.log(
    `epoch ${String(epoch).padStart(2)}  loss ${(loss / batches).toFixed(3)}  test acc ${(testAcc * 100).toFixed(2)}%`,
  )
}

// Round weights to 4 significant figures to keep the bundled JSON small, then
// re-measure so the accuracy we ship is the accuracy of the rounded weights
// that actually run in the browser — not the fuller-precision training copy.
for (const p of net.params()) {
  const v = p.value
  for (let i = 0; i < v.data.length; i++) v.data[i] = Number(v.data[i].toPrecision(4))
}

const finalAcc = accuracy(net.predict(testX), testY)
const out = { ...net.serialize(), testAccuracy: finalAcc }
const path = join(here, '..', 'src', 'weights', 'digits.json')
writeFileSync(path, JSON.stringify(out))
console.log(`\nfinal test accuracy ${(finalAcc * 100).toFixed(2)}% (rounded weights)  →  wrote ${path}`)
