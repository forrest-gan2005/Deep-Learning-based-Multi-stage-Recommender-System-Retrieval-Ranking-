# RecEngine-DeepFM: High-Performance Recommendation Engine

A production-ready recommendation system implementation featuring **DeepFM** (Deep Factorization Machines). This project covers the entire machine learning lifecycle: from offline training in **Python/PyTorch** to high-performance online inference in **C++ (LibTorch)**.

## Key Features
- **Algorithm**: Hybrid DeepFM model capturing both low-order and high-order feature interactions.
- **Performance**: Sub-millisecond inference latency powered by C++ and LibTorch.
- **Cross-Platform**: Seamless model transition using **TorchScript** tracing.
- **Hardware Acceleration**: Full support for Apple Silicon (MPS) during training.

## Project Structure
```text
├── data/raw/ml-1m/          # MovieLens 1M dataset
├── src/
│   ├── python/              # Training pipeline (Preprocessing, Dataset, Training)
│   │   ├── models/          # DeepFM architecture implementation
│   │   └── export_model.py  # TorchScript serialization script
│   └── cpp/                 # Inference engine (C++ source)
├── CMakeLists.txt           # C++ build configuration
├── deepfm_traced.pt         # Serialized model for C++ runtime
└── requirements.txt         # Python dependencies
```
## Getting Started
### 1. Requirements
Python 3.9+

PyTorch 2.0+

CMake 3.10+

LibTorch (macOS: brew install pytorch)

### 2. Training (Python)
First, download the MovieLens 1M dataset into data/raw/ml-1m/, then run:

Bash
```
# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/python

# Run training and export the model
python src/python/train.py
python src/python/export_model.py
This will generate deepfm_model.pth and deepfm_traced.pt.
```

### 3. Inference (C++)
Build the high-performance inference executable:

Bash
```
mkdir build && cd build
cmake ..
make
./deepfm_inference
```
## Performance & Results
Training Loss: Successfully converged from ~1.54 to 0.56 on MovieLens 1M.

Inference: Demonstrated Zero-Copy tensor creation using torch::from_blob in C++ for maximum throughput.

## Implementation 
DetailsFM Component: Efficiently implemented using the $O(kn)$ complexity reduction formula:
$$\frac{1}{2} \sum_{f=1}^k \left( (\sum_{i=1}^n v_{i,f}x_i)^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2 \right)$$
Deep Component: 3-layer MLP (128 -> 64 -> 1) for capturing non-linear high-order features.
Inference Bridge: Utilizing torch::jit (TorchScript) to eliminate Python runtime overhead.
