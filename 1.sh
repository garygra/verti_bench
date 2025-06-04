#!/bin/bash

# File types tracked by LFS
PATTERNS=("*.obj" "*.mtl" "*.dae" "*.usd" "*.usda" "*.psd" "*.hdr" "*.gif" "*.mp4" "*.pt" "*.jit" "*.hdf5" "*.torch" "*.tar.bz2")

echo "Removing matching files from Git index..."

for pattern in "${PATTERNS[@]}"; do
  find . -type f -name "$pattern" -print0 | xargs -0 git rm --cached --ignore-unmatch
done

echo "Re-adding files to apply LFS tracking..."
git add .

echo "Committing..."
git commit -m "Re-track all matching files via Git LFS"

echo "✅ Done. Now push with:"
echo "   git push origin main"
