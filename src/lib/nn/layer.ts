import { Matrix } from './matrix'

/** A learnable tensor plus the gradient the backward pass writes into it. */
export type Param = { value: Matrix; grad: Matrix }

/**
 * Every layer can run a forward pass, then push a gradient back through itself.
 * Layers cache whatever the backward pass needs (their input, their output…).
 */
export interface Layer {
  forward(x: Matrix): Matrix
  backward(gradOut: Matrix): Matrix
  params(): Param[]
}

export type Activation = 'relu' | 'tanh' | 'sigmoid'

/**
 * Fully-connected layer: y = x · W + b.
 * Weight init is scaled to the activation that follows (He for ReLU, Xavier
 * for the saturating ones) so signals don't blow up or vanish at depth.
 */
export class Dense implements Layer {
  W: Param
  b: Param
  private input: Matrix | null = null

  constructor(nIn: number, nOut: number, activation: Activation) {
    const std = activation === 'relu' ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn)
    this.W = { value: Matrix.randn(nIn, nOut, std), grad: Matrix.zeros(nIn, nOut) }
    this.b = { value: Matrix.zeros(1, nOut), grad: Matrix.zeros(1, nOut) }
  }

  forward(x: Matrix): Matrix {
    this.input = x
    return x.matmul(this.W.value).addRowVector(this.b.value)
  }

  backward(gradOut: Matrix): Matrix {
    const x = this.input!
    // dW = xᵀ · gradOut ; db = column-sum of gradOut ; dx = gradOut · Wᵀ
    this.W.grad = x.transpose().matmul(gradOut)
    this.b.grad = gradOut.sumRows()
    return gradOut.matmul(this.W.value.transpose())
  }

  params(): Param[] {
    return [this.W, this.b]
  }
}

/** ReLU: max(0, x). Gradient flows only where the input was positive. */
export class ReLU implements Layer {
  private mask: Matrix | null = null
  forward(x: Matrix): Matrix {
    this.mask = x.map((v) => (v > 0 ? 1 : 0))
    return x.map((v) => (v > 0 ? v : 0))
  }
  backward(gradOut: Matrix): Matrix {
    return gradOut.hadamard(this.mask!)
  }
  params(): Param[] {
    return []
  }
}

/** tanh, with the classic 1 − tanh² local gradient. */
export class Tanh implements Layer {
  private out: Matrix | null = null
  forward(x: Matrix): Matrix {
    this.out = x.map(Math.tanh)
    return this.out
  }
  backward(gradOut: Matrix): Matrix {
    return gradOut.hadamard(this.out!.map((y) => 1 - y * y))
  }
  params(): Param[] {
    return []
  }
}

/** Logistic sigmoid, local gradient σ(1 − σ). */
export class Sigmoid implements Layer {
  private out: Matrix | null = null
  forward(x: Matrix): Matrix {
    this.out = x.map((v) => 1 / (1 + Math.exp(-v)))
    return this.out
  }
  backward(gradOut: Matrix): Matrix {
    return gradOut.hadamard(this.out!.map((y) => y * (1 - y)))
  }
  params(): Param[] {
    return []
  }
}

export function makeActivation(kind: Activation): Layer {
  if (kind === 'relu') return new ReLU()
  if (kind === 'tanh') return new Tanh()
  return new Sigmoid()
}
