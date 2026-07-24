/**
 * Correctness tests for the from-scratch neural-network engine.
 *
 * These are fully deterministic — weights are set to fixed values, so there is
 * no random initialisation and nothing can flake in CI. The convergence proof
 * (does it actually learn?) lives in scripts/sanity.ts, which is inherently
 * stochastic and belongs in a manual run, not a gate.
 *
 * The gradient check is the one that matters. It compares the gradient computed
 * by backpropagation against a numerical gradient from finite differences. If
 * backprop has a bug — a missing transpose, a wrong chain-rule term — these two
 * disagree. It is the standard way to prove an autodiff implementation is
 * correct, and it is the test an interviewer would ask for.
 */

import { describe, expect, it } from 'vitest'
import { Matrix } from './matrix'
import { softmax, softmaxCrossEntropy, accuracy } from './losses'
import { Network } from './network'

describe('Matrix', () => {
  it('multiplies matrices correctly', () => {
    // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
    const a = Matrix.from([
      [1, 2],
      [3, 4],
    ])
    const b = Matrix.from([
      [5, 6],
      [7, 8],
    ])
    const c = a.matmul(b)
    expect(c.get(0, 0)).toBe(19)
    expect(c.get(0, 1)).toBe(22)
    expect(c.get(1, 0)).toBe(43)
    expect(c.get(1, 1)).toBe(50)
  })

  it('transposes', () => {
    const m = Matrix.from([[1, 2, 3]])
    const t = m.transpose()
    expect(t.rows).toBe(3)
    expect(t.cols).toBe(1)
    expect(t.get(2, 0)).toBe(3)
  })

  it('matmul respects inner dimensions', () => {
    const a = Matrix.from([[1, 2, 3]]) // 1x3
    const b = Matrix.from([[1], [1], [1]]) // 3x1
    const c = a.matmul(b) // 1x1
    expect(c.rows).toBe(1)
    expect(c.cols).toBe(1)
    expect(c.get(0, 0)).toBe(6)
  })
})

describe('softmax', () => {
  it('produces a probability distribution per row', () => {
    const logits = Matrix.from([
      [1, 2, 3],
      [0, 0, 0],
    ])
    const p = softmax(logits)
    for (let i = 0; i < p.rows; i++) {
      let sum = 0
      for (let j = 0; j < p.cols; j++) {
        expect(p.get(i, j)).toBeGreaterThan(0)
        expect(p.get(i, j)).toBeLessThan(1)
        sum += p.get(i, j)
      }
      expect(sum).toBeCloseTo(1, 10)
    }
  })

  it('is monotonic in the logits', () => {
    const p = softmax(Matrix.from([[1, 2, 3]]))
    expect(p.get(0, 2)).toBeGreaterThan(p.get(0, 1))
    expect(p.get(0, 1)).toBeGreaterThan(p.get(0, 0))
  })

  it('is shift-invariant (numerical stability)', () => {
    const a = softmax(Matrix.from([[1, 2, 3]]))
    const b = softmax(Matrix.from([[1001, 1002, 1003]]))
    for (let j = 0; j < 3; j++) expect(a.get(0, j)).toBeCloseTo(b.get(0, j), 10)
  })
})

describe('softmaxCrossEntropy', () => {
  it('is near zero for a confident, correct prediction', () => {
    const logits = Matrix.from([[10, 0, 0]])
    const { loss } = softmaxCrossEntropy(logits, [0])
    expect(loss).toBeLessThan(0.01)
  })

  it('is large for a confident, wrong prediction', () => {
    const logits = Matrix.from([[10, 0, 0]])
    const { loss } = softmaxCrossEntropy(logits, [2])
    expect(loss).toBeGreaterThan(5)
  })

  it('equals ln(C) for uniform logits over C classes', () => {
    const logits = Matrix.from([[0, 0, 0]])
    const { loss } = softmaxCrossEntropy(logits, [1])
    expect(loss).toBeCloseTo(Math.log(3), 6)
  })
})

describe('accuracy', () => {
  it('scores argmax predictions', () => {
    const probs = Matrix.from([
      [0.7, 0.2, 0.1], // -> 0
      [0.1, 0.1, 0.8], // -> 2
    ])
    expect(accuracy(probs, [0, 2])).toBe(1)
    expect(accuracy(probs, [1, 2])).toBe(0.5)
  })
})

describe('backpropagation — gradient check', () => {
  // Fully deterministic: build a net, then overwrite every weight with a fixed
  // pattern so there is no randomness left. The finite-difference check must
  // then match backprop to within numerical tolerance for the run to pass.
  function fixedNet(): Network {
    const net = Network.mlp([3, 4, 3], 'tanh')
    let k = 0
    for (const p of net.params()) {
      const { value } = p
      for (let i = 0; i < value.rows; i++) {
        for (let j = 0; j < value.cols; j++) {
          // small, spread, sign-varying — keeps tanh out of saturation
          value.set(i, j, (((k++ % 7) - 3) * 0.13))
        }
      }
    }
    return net
  }

  const X = Matrix.from([
    [0.5, -0.2, 0.1],
    [-0.3, 0.8, -0.6],
  ])
  const labels = [2, 0]

  function lossOf(net: Network): number {
    return softmaxCrossEntropy(net.forward(X), labels).loss
  }

  it('analytic gradients match finite differences', () => {
    const net = fixedNet()

    // analytic gradients via backprop
    const { grad } = softmaxCrossEntropy(net.forward(X), labels)
    net.backward(grad)

    const eps = 1e-5
    let maxDiff = 0
    let checked = 0

    for (const p of net.params()) {
      const { value, grad: analytic } = p
      // sample a few entries per parameter tensor (checking all is overkill)
      for (let i = 0; i < value.rows; i++) {
        for (let j = 0; j < value.cols; j += 2) {
          const original = value.get(i, j)

          value.set(i, j, original + eps)
          const lossPlus = lossOf(net)
          value.set(i, j, original - eps)
          const lossMinus = lossOf(net)
          value.set(i, j, original) // restore

          const numeric = (lossPlus - lossMinus) / (2 * eps)
          const diff = Math.abs(numeric - analytic.get(i, j))
          maxDiff = Math.max(maxDiff, diff)
          checked += 1
        }
      }
    }

    expect(checked).toBeGreaterThan(10)
    // centered differences on this scale agree with correct backprop to ~1e-6;
    // 1e-4 is a comfortable, non-flaky bound.
    expect(maxDiff).toBeLessThan(1e-4)
  })
})
