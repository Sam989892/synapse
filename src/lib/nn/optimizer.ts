import { Matrix } from './matrix'
import type { Param } from './layer'

export interface Optimizer {
  step(params: Param[]): void
  lr: number
}

/** Plain stochastic gradient descent with optional momentum. */
export class SGD implements Optimizer {
  lr: number
  private momentum: number
  private velocity = new WeakMap<Param, Matrix>()

  constructor(lr: number, momentum = 0.9) {
    this.lr = lr
    this.momentum = momentum
  }

  step(params: Param[]): void {
    for (const p of params) {
      let v = this.velocity.get(p)
      if (!v) {
        v = Matrix.zeros(p.value.rows, p.value.cols)
        this.velocity.set(p, v)
      }
      for (let k = 0; k < p.value.data.length; k++) {
        v.data[k] = this.momentum * v.data[k] - this.lr * p.grad.data[k]
        p.value.data[k] += v.data[k]
      }
    }
  }
}

/**
 * Adam — adaptive moment estimation. Keeps a running mean (m) and variance (v)
 * of each gradient and normalises the step, which makes training far less
 * sensitive to the learning-rate choice.
 */
export class Adam implements Optimizer {
  lr: number
  private beta1 = 0.9
  private beta2 = 0.999
  private eps = 1e-8
  private t = 0
  private m = new WeakMap<Param, Matrix>()
  private v = new WeakMap<Param, Matrix>()

  constructor(lr: number) {
    this.lr = lr
  }

  step(params: Param[]): void {
    this.t++
    const bc1 = 1 - Math.pow(this.beta1, this.t)
    const bc2 = 1 - Math.pow(this.beta2, this.t)
    for (const p of params) {
      let m = this.m.get(p)
      let v = this.v.get(p)
      if (!m) {
        m = Matrix.zeros(p.value.rows, p.value.cols)
        this.m.set(p, m)
      }
      if (!v) {
        v = Matrix.zeros(p.value.rows, p.value.cols)
        this.v.set(p, v)
      }
      for (let k = 0; k < p.value.data.length; k++) {
        const g = p.grad.data[k]
        m.data[k] = this.beta1 * m.data[k] + (1 - this.beta1) * g
        v.data[k] = this.beta2 * v.data[k] + (1 - this.beta2) * g * g
        const mHat = m.data[k] / bc1
        const vHat = v.data[k] / bc2
        p.value.data[k] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps)
      }
    }
  }
}
