# Synapse — a neural network from scratch

[![CI](https://github.com/Sam989892/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/Sam989892/synapse/actions/workflows/ci.yml)

A working neural network built from first principles in TypeScript. **No TensorFlow, no PyTorch, no NumPy, no API** — every matrix multiply, activation, loss gradient and the entire backpropagation pass is hand-written. It runs 100% in the browser.

Two things prove the engine is real:

1. **A live training playground** — pick a dataset (spiral, circles, XOR, moons, gaussian), shape the network, hit Train, and watch the decision boundary morph and the loss fall in real time. Every frame runs genuine forward + backward passes.
2. **A handwritten-digit recognizer** — draw a digit and the same engine reads it. The weights were trained on MNIST *offline using this very engine*, then saved and loaded for in-browser inference.

**Stack:** TypeScript · React 19 · Vite · Tailwind CSS v4 · Framer Motion. Zero machine-learning libraries.

## The engine (`src/lib/nn/`, ~500 lines)

| File | What it implements |
|------|--------------------|
| `matrix.ts` | Float64Array-backed matrix: matmul, transpose, row-broadcast, Gaussian (Box–Muller) init |
| `layer.ts` | Dense layer forward/backward; ReLU, tanh, sigmoid with their local gradients; He/Xavier init |
| `losses.ts` | Numerically-stable softmax and the fused softmax-cross-entropy gradient |
| `optimizer.ts` | SGD with momentum, and Adam (bias-corrected first/second moments) |
| `network.ts` | Sequential model: forward, backprop, train step, serialize/deserialize |

The maths depends on nothing but `Math`. This is what a deep-learning framework does, in miniature, readable end to end.

## Verified

- `npm test` — **11 deterministic unit tests** (weights fixed, nothing random), including a **gradient check**: the gradient from backpropagation is compared against a numerical gradient from finite differences and must agree to within `1e-4`. This is the standard proof that an autodiff implementation is correct — if a chain-rule term or a transpose were wrong, the two disagree. Also covers matmul, softmax (distribution + shift-invariance), and cross-entropy.
- `npx tsx scripts/sanity.ts` — trains spiral / XOR / circles / moons / gaussian to **99–100%** accuracy (the convergence proof; stochastic, so it's a script not a CI gate).
- `npx tsx scripts/train-digits.ts` — trains the 784→64→32→10 digit model on MNIST to **94.25%** held-out test accuracy and exports the weights.

## Run it

```bash
npm install
npm run dev        # http://localhost:5176
npm test           # 11 tests incl. the backprop gradient check
npm run build

# reproduce the proofs
npx tsx scripts/sanity.ts
npx tsx scripts/train-digits.ts    # retrains + re-exports src/weights/digits.json
```

## Deploy (Vercel)

Static SPA — push to GitHub, import at vercel.com, no config, no environment variables, no backend.

## Why this is on my CV

Most "AI" portfolio projects call a hosted model behind an API. This one implements the model. It shows I understand what backpropagation actually computes, how gradients flow through layers, why Adam converges where SGD stalls, and how a softmax-cross-entropy gradient collapses to `(p − y)`. The interactive demos make that legible to anyone in ten seconds.

### Resume bullets

- Built a neural-network engine from scratch in TypeScript — matrix ops, dense layers, ReLU/tanh/sigmoid, softmax-cross-entropy, SGD and Adam, and full backpropagation — with no ML libraries, and trained it to 94% on MNIST
- Shipped it as an interactive browser app: a real-time training playground that visualises the decision boundary, and a handwritten-digit recognizer running inference client-side

---

Concept project — designed & built by Saiyed (Sam) Madni.
