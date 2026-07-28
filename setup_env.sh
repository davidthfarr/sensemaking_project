#!/bin/bash
echo "Setting up environment..."

pip install --force-reinstall --no-deps \
  torch==2.4.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121 -q

pip install --force-reinstall --no-deps \
  numpy==1.26.4 \
  tokenizers==0.21.0 \
  safetensors==0.4.3 \
  transformers==4.47.0 \
  sentence-transformers==3.3.0 -q

pip install -e . --no-deps -q

echo "Verifying..."
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
from transformers import AutoTokenizer
print('Transformers: OK')
from sentence_transformers import SentenceTransformer
print('SentenceTransformers: OK')
"
echo "Setup complete."
