import { Matrix } from './matrix'

/** Row-wise softmax, shifted by the row max for numerical stability. */
export function softmax(logits: Matrix): Matrix {
  const out = new Matrix(logits.rows, logits.cols)
  for (let i = 0; i < logits.rows; i++) {
    let max = -Infinity
    for (let j = 0; j < logits.cols; j++) max = Math.max(max, logits.get(i, j))
    let sum = 0
    for (let j = 0; j < logits.cols; j++) {
      const e = Math.exp(logits.get(i, j) - max)
      out.set(i, j, e)
      sum += e
    }
    for (let j = 0; j < logits.cols; j++) out.set(i, j, out.get(i, j) / sum)
  }
  return out
}

/**
 * Softmax + cross-entropy, fused. The fused gradient is simply
 * (probs − oneHot) / batchSize — clean, stable, and cheap.
 */
export function softmaxCrossEntropy(
  logits: Matrix,
  labels: number[],
): { loss: number; grad: Matrix; probs: Matrix } {
  const probs = softmax(logits)
  const n = logits.rows
  let loss = 0
  const grad = probs.clone()
  for (let i = 0; i < n; i++) {
    const y = labels[i]
    loss -= Math.log(Math.max(probs.get(i, y), 1e-12))
    grad.set(i, y, grad.get(i, y) - 1) // subtract the one-hot target
  }
  return { loss: loss / n, grad: grad.scale(1 / n), probs }
}

/** Fraction of rows whose arg-max prediction matches the label. */
export function accuracy(probs: Matrix, labels: number[]): number {
  let correct = 0
  for (let i = 0; i < probs.rows; i++) {
    let best = 0
    let bestVal = -Infinity
    for (let j = 0; j < probs.cols; j++) {
      const v = probs.get(i, j)
      if (v > bestVal) {
        bestVal = v
        best = j
      }
    }
    if (best === labels[i]) correct++
  }
  return correct / probs.rows
}
