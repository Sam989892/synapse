// Quick proof the from-scratch engine actually learns. Run: npx tsx scripts/sanity.ts
import { Matrix, Network, Adam, accuracy } from '../src/lib/nn'
import { makeDataset, type DatasetId } from '../src/lib/datasets'

function run(id: DatasetId) {
  const data = makeDataset(id, 400, 0.1)
  const X = Matrix.from(data.X)
  const net = Network.mlp([2, 16, 16, 2], 'tanh')
  const opt = new Adam(0.03)
  for (let epoch = 0; epoch < 400; epoch++) net.trainStep(X, data.y, opt)
  const acc = accuracy(net.predict(X), data.y)
  console.log(`${id.padEnd(9)} accuracy: ${(acc * 100).toFixed(1)}%`)
  return acc
}

const results = (['gaussian', 'xor', 'circles', 'moons', 'spiral'] as DatasetId[]).map(run)
const ok = results.every((a) => a > 0.85)
console.log(ok ? '\nPASS — the network learns every dataset.' : '\nFAIL — something is off.')
process.exit(ok ? 0 : 1)
