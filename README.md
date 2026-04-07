# Micro GPT

A minimal character-level generative pre-trained transformer built from scratch in PyTorch. This project implements the core transformer architecture — self-attention, multi-head attention, feedforward layers, and positional embeddings — to train and run a small language model capable of generating text.

## Architecture

The model follows a decoder-only transformer design:

```
Input Tokens
    │
    ▼
┌──────────────────────┐
│  Token Embedding     │
│  + Position Embedding│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Transformer Block   │ ×N
│  ├─ Multi-Head       │
│  │  Self-Attention   │
│  ├─ Layer Norm       │
│  ├─ Feed Forward     │
│  └─ Layer Norm       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Linear → Logits     │
└──────────────────────┘
```

**Default hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Embedding size | 384 |
| Attention heads | 8 (training) / 1 (inference) |
| Transformer blocks | 1 |
| Context length | 128 tokens |
| Dropout | 0.2 |
| Learning rate | 3e-4 |

## Project Structure

```
micro-gpt/
├── training.py              # Model training script
├── model.py                 # Model definition and inference CLI
├── GPT-v1.ipynb             # Jupyter notebook for experimentation
├── DataExtractors/          # Scripts for preparing training data
│   ├── dataExtract-v1.py    # OpenWebText extraction (single-pass)
│   └── dataExtract-v2.py    # OpenWebText extraction (multi-file split)
└── vocab.txt                # Character vocabulary built from training data
```

### Key Files

- **`training.py`** — Trains the GPT model end-to-end. Includes data loading via memory-mapped files, batch sampling, loss estimation, and model checkpointing. Run with `-batch_size` to configure.
- **`model.py`** — Contains all model architecture classes (`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPTLanguageModel`) and an interactive generation loop. Run directly for a command-line inference prompt.
- **`DataExtractors/`** — Utilities for extracting and decompressing OpenWebText `.xz` archives into plain text for training.

## Setup

This project was developed on an Apple M1 chip running MacOS Sonoma 14.6.1

Development made use of a virtual environment for training and GPU processing.

Because this project was developed in MacOS, Cuda was not utilized.

Instead, mps was used. (more on this later)

To setup the development environment identical to the one in which this LLM was created, follow the following commands:

1. Create a Virtual Environment
- **python -m venv VirtualEnv**
- This creates a virtual environment named VirtualEnv.
- A virtual environment isolates dependencies from the system Python installation, ensuring your project has a controlled environment.
- Be sure to activate the virtual environment using **source VirtualEnv/bin/activate**.

2. Install Required Python Libraries
- **pip install matplotlib numpy pylzma ipykernel jupyter**
- matplotlib → A plotting library for visualization.
- numpy → Provides numerical operations and support for large, multi-dimensional arrays.
- pylzma → A Python library for handling LZMA (compression).
- ipykernel → Allows Jupyter to run different Python environments (kernels).
- jupyter → Installs Jupyter Notebook, a web-based interface for running Python code interactively.
- Note we will not wake use of pylzma due to compatability issues and its existense as a python standard library import, **import lzma**.
  
3. Install PyTorch and Related Libraries
- **pip3 install torch torchvision torchaudio**
- torch → The core PyTorch library for tensor computations and deep learning.
- torchvision → Provides image datasets and pre-processing utilities.
- torchaudio → Provides audio processing tools.
- PyTorch will automatically detect the best backend for execution.
- Since you're on an M1 Mac, PyTorch should install the mps backend instead of CUDA.
  
4. Create a Jupyter Kernel for the Virtual Environment
- **python -m ipykernel install --user --name=gpu_kernel --display-name "gpu kernel"**
- Creates a Jupyter kernel named gpu_kernel using the virtual environment.
- --display-name "gpu kernel" sets how the kernel appears in Jupyter Notebook.
- After this, you can open Jupyter and select "gpu kernel" as your notebook kernel.

Once completed your machine should be ready for testing and development.

To launch jupyter, use the command **jupyter notebook**.

## Usage

### Training

```bash
python training.py -batch_size 64
```

This trains the model on the data referenced in the training script, evaluates loss periodically, and saves the final weights to `model-01.pkl`.

### Inference

```bash
python model.py
```

Loads `model-01.pkl` and `vocab.txt`, then enters an interactive loop where you can type prompts and receive generated continuations.

```
Prompt:
The meaning of life is
Completion:
The meaning of life is to be found in the pursuit of ...
```

## How It Works

1. **Tokenization** — Character-level encoding maps each unique character in `vocab.txt` to an integer.
2. **Embedding** — Token indices are converted to dense vectors and combined with learned positional embeddings.
3. **Self-Attention** — Each token attends to all previous tokens (causal masking prevents attending to future tokens).
4. **Feed Forward** — A two-layer MLP with ReLU activation processes each token independently after attention.
5. **Generation** — At inference time, the model autoregressively samples one token at a time, appending each prediction to the context.

## Compute

The model uses Apple's Metal Performance Shaders (MPS) backend when running on Apple Silicon. On other systems, it falls back to CPU. The device is selected automatically:

```python
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

## License

This project is provided as-is for educational purposes.
