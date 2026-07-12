/**
 * Matrix — a tiny dense linear-algebra core, backed by a flat Float64Array.
 * Everything the neural network needs is built on these operations.
 * No math library, no dependencies. Row-major storage: element (i, j) lives at
 * index i * cols + j.
 */
export class Matrix {
  readonly rows: number
  readonly cols: number
  readonly data: Float64Array

  constructor(rows: number, cols: number, data?: Float64Array) {
    this.rows = rows
    this.cols = cols
    this.data = data ?? new Float64Array(rows * cols)
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j]
  }

  set(i: number, j: number, v: number): void {
    this.data[i * this.cols + j] = v
  }

  /** Build from a nested JS array. */
  static from(rows2d: number[][]): Matrix {
    const rows = rows2d.length
    const cols = rows2d[0].length
    const m = new Matrix(rows, cols)
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) m.data[i * cols + j] = rows2d[i][j]
    return m
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols)
  }

  /**
   * Gaussian-initialised weights, scaled by `std`. Uses the Box–Muller
   * transform so we depend on nothing but Math.random.
   */
  static randn(rows: number, cols: number, std: number): Matrix {
    const m = new Matrix(rows, cols)
    for (let k = 0; k < m.data.length; k++) m.data[k] = gaussian() * std
    return m
  }

  clone(): Matrix {
    return new Matrix(this.rows, this.cols, this.data.slice())
  }

  /** Matrix product: (rows x k) · (k x cols) → (rows x cols). */
  matmul(b: Matrix): Matrix {
    if (this.cols !== b.rows)
      throw new Error(`matmul shape mismatch: ${this.shape()} · ${b.shape()}`)
    const out = new Matrix(this.rows, b.cols)
    const k = this.cols
    for (let i = 0; i < this.rows; i++) {
      for (let p = 0; p < k; p++) {
        const a = this.data[i * k + p]
        if (a === 0) continue
        const bRow = p * b.cols
        const oRow = i * b.cols
        for (let j = 0; j < b.cols; j++) out.data[oRow + j] += a * b.data[bRow + j]
      }
    }
    return out
  }

  transpose(): Matrix {
    const out = new Matrix(this.cols, this.rows)
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++) out.data[j * this.rows + i] = this.data[i * this.cols + j]
    return out
  }

  /** Add a (1 x cols) bias vector to every row. Returns a new matrix. */
  addRowVector(bias: Matrix): Matrix {
    const out = new Matrix(this.rows, this.cols)
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++)
        out.data[i * this.cols + j] = this.data[i * this.cols + j] + bias.data[j]
    return out
  }

  /** Column-wise sum → (1 x cols). Used to reduce a batch of gradients to the bias gradient. */
  sumRows(): Matrix {
    const out = new Matrix(1, this.cols)
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++) out.data[j] += this.data[i * this.cols + j]
    return out
  }

  map(fn: (v: number) => number): Matrix {
    const out = new Matrix(this.rows, this.cols)
    for (let k = 0; k < this.data.length; k++) out.data[k] = fn(this.data[k])
    return out
  }

  /** Element-wise (Hadamard) product. */
  hadamard(b: Matrix): Matrix {
    const out = new Matrix(this.rows, this.cols)
    for (let k = 0; k < this.data.length; k++) out.data[k] = this.data[k] * b.data[k]
    return out
  }

  scale(s: number): Matrix {
    return this.map((v) => v * s)
  }

  shape(): string {
    return `${this.rows}x${this.cols}`
  }
}

/** One standard-normal sample via Box–Muller. */
function gaussian(): number {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
