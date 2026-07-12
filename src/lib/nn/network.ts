import { Matrix } from './matrix'
import { Dense, makeActivation, type Activation, type Layer, type Param } from './layer'
import { softmax, softmaxCrossEntropy, accuracy } from './losses'
import type { Optimizer } from './optimizer'

/**
 * A sequential stack of layers ending in raw logits. Classification head
 * (softmax) lives in the loss, so `forward` returns logits and `predict`
 * returns probabilities.
 */
export class Network {
  layers: Layer[] = []

  /** Build an MLP: [inputDim, ...hiddenDims, outputDim]. */
  static mlp(dims: number[], activation: Activation): Network {
    const net = new Network()
    for (let i = 0; i < dims.length - 1; i++) {
      net.layers.push(new Dense(dims[i], dims[i + 1], activation))
      if (i < dims.length - 2) net.layers.push(makeActivation(activation)) // no activation on the logits
    }
    return net
  }

  forward(x: Matrix): Matrix {
    let out = x
    for (const layer of this.layers) out = layer.forward(out)
    return out
  }

  backward(grad: Matrix): void {
    let g = grad
    for (let i = this.layers.length - 1; i >= 0; i--) g = this.layers[i].backward(g)
  }

  params(): Param[] {
    return this.layers.flatMap((l) => l.params())
  }

  /** Class probabilities for a batch. */
  predict(x: Matrix): Matrix {
    return softmax(this.forward(x))
  }

  /**
   * One optimisation step over a batch: forward → loss → backward → update.
   * Returns the loss and accuracy on that batch.
   */
  trainStep(x: Matrix, labels: number[], opt: Optimizer): { loss: number; acc: number } {
    const logits = this.forward(x)
    const { loss, grad, probs } = softmaxCrossEntropy(logits, labels)
    this.backward(grad)
    opt.step(this.params())
    return { loss, acc: accuracy(probs, labels) }
  }

  /** Flatten every weight and bias to plain arrays — used to save/load models. */
  serialize(): SerializedNet {
    const layers: SerializedLayer[] = []
    for (const l of this.layers) {
      if (l instanceof Dense) {
        layers.push({
          type: 'dense',
          nIn: l.W.value.rows,
          nOut: l.W.value.cols,
          W: Array.from(l.W.value.data),
          b: Array.from(l.b.value.data),
        })
      } else {
        layers.push({ type: 'activation', kind: activationName(l) })
      }
    }
    return { layers }
  }

  static deserialize(data: SerializedNet): Network {
    const net = new Network()
    for (const l of data.layers) {
      if (l.type === 'dense') {
        const dense = new Dense(l.nIn, l.nOut, 'relu')
        dense.W.value.data.set(l.W)
        dense.b.value.data.set(l.b)
        net.layers.push(dense)
      } else {
        net.layers.push(makeActivation(l.kind))
      }
    }
    return net
  }
}

export type SerializedLayer =
  | { type: 'dense'; nIn: number; nOut: number; W: number[]; b: number[] }
  | { type: 'activation'; kind: Activation }

export type SerializedNet = { layers: SerializedLayer[] }

function activationName(l: Layer): Activation {
  const n = l.constructor.name
  if (n === 'Tanh') return 'tanh'
  if (n === 'Sigmoid') return 'sigmoid'
  return 'relu'
}
