#!/usr/bin/env bash
# Downloads the real full MNIST dataset (60k train / 10k test) into
# scripts/mnist-data/. Files come from the CVDF mirror (Google-hosted copy of
# the original LeCun IDX files). Run once before scripts/train-digits.ts.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mnist-data"
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
mkdir -p "$DIR"
cd "$DIR"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "$f" ]; then
    echo "fetching $f ..."
    curl -sSL -o "$f.gz" "$BASE/$f.gz"
    gunzip -f "$f.gz"
  fi
done
echo "MNIST ready in $DIR"
