#!/usr/bin/env bash
# ============================================================
# Fetch H&M Personalized Fashion Recommendations dataset
# Run this from the ROOT of your project (where products.csv etc. live)
# ============================================================
set -euo pipefail

RAW_DIR="data/hm_raw"
mkdir -p "$RAW_DIR"

# 1. Kaggle CLI (skip if already installed)
pip install --quiet kaggle

# 2. Kaggle credentials
#    - Go to kaggle.com -> Settings -> API -> "Create New Token"
#    - This downloads kaggle.json. Put it here:
mkdir -p ~/.kaggle
echo "-> If you haven't already, copy your kaggle.json into ~/.kaggle/kaggle.json now."
echo "   Then re-run this script."
if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found. Aborting." >&2
  exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

# 3. IMPORTANT one-time manual step: you must accept the competition rules
#    in your browser before the API will let you download:
#    https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/rules
echo "-> Make sure you've clicked 'I Understand and Accept' on the rules page above."

# 4. Download the three metadata files (small) + the images archive (~25GB, full-size)
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f articles.csv -p "$RAW_DIR"
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f customers.csv -p "$RAW_DIR"
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv -p "$RAW_DIR"
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f images.zip -p "$RAW_DIR"

# 5. Unzip the metadata CSVs (kaggle wraps each -f download in its own zip)
cd "$RAW_DIR"
for z in articles.csv.zip customers.csv.zip transactions_train.csv.zip; do
  [ -f "$z" ] && unzip -o "$z" && rm "$z"
done
cd - > /dev/null

echo "Done. Raw files are in $RAW_DIR/ (articles.csv, customers.csv, transactions_train.csv, images.zip)"
echo "NOTE: images.zip is left zipped on purpose — the transform script in step 2"
echo "extracts only the sampled product images it needs directly from the zip,"
echo "so you never have to unpack all ~25GB."
