/**
 * Trains the handwritten-digit model OFFLINE using the same from-scratch engine
 * the browser uses for inference. Nothing here is an external service — the
 * `mnist` npm package just bundles the pixel data. Output: src/weights/digits.json.
 *
 * Run: npx tsx scripts/train-digits.ts
 */
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import mnist from 'mnist'
import { Matrix, Network, Adam, accuracy } from '../src/lib/nn'

const TRAIN = 8000
const TEST = 2000
const EPOCHS = 20
const BATCH = 64
const LR = 0.002

const here = dirname(fileURLToPath(import.meta.url))

type Sample = { input: number[]; output: number[] }
const oneHotToLabel = (o: number[]) => o.indexOf(Math.max(...o))

const set = mnist.set(TRAIN, TEST)
const trainX = set.training.map((s: Sample) => s.input)
const trainY = set.training.map((s: Sample) => oneHotToLabel(s.output))
const testX = Matrix.from(set.test.map((s: Sample) => s.input))
const testY = set.test.map((s: Sample) => oneHotToLabel(s.output))

// 784 → 64 → 32 → 10 multilayer perceptron.
const net = Network.mlp([784, 64, 32, 10], 'relu')
const opt = new Adam(LR)

const idx = trainX.map((_, i) => i)
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  // shuffle each epoch
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  let loss = 0
  let batches = 0
  for (let b = 0; b < idx.length; b += BATCH) {
    const slice = idx.slice(b, b + BATCH)
    const xb = Matrix.from(slice.map((k) => trainX[k]))
    const yb = slice.map((k) => trainY[k])
    loss += net.trainStep(xb, yb, opt).loss
    batches++
  }
  const testAcc = accuracy(net.predict(testX), testY)
  console.log(
    `epoch ${String(epoch).padStart(2)}  loss ${(loss / batches).toFixed(3)}  test acc ${(testAcc * 100).toFixed(2)}%`,
  )
}

const finalAcc = accuracy(net.predict(testX), testY)
const out = { ...net.serialize(), testAccuracy: finalAcc }
const path = join(here, '..', 'src', 'weights', 'digits.json')
writeFileSync(path, JSON.stringify(out))
console.log(`\nSaved ${path} — final test accuracy ${(finalAcc * 100).toFixed(2)}%`)
