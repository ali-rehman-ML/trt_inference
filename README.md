# TRT Inference

A Python package for creating TensorRT engines from ONNX models and performing inference with dynamic shapes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ali-rehman-ML/trt_inference.git
   cd trt_inference
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package:
   ```bash
   pip install .
   ```

## Usage

### Creating a TensorRT Engine

```python
from trt_inference import EngineBuilder

builder = EngineBuilder(verbose=True, workspace=16)
builder.create_network("resnet18.onnx")
builder.create_engine("resnet18.trt", precision="fp16")
```

### Performing Inference

```python
import numpy as np
from trt_inference import TRTInference

trt_inference = TRTInference("resnet18.trt", max_batch_size=32)
input_data = np.random.randn(4, 3, 224, 224).astype(np.float32)
outputs = trt_inference.infer(input_data)
print(f"Output shapes: {[output.shape for output in outputs]}")
```

See the `examples/` directory for more examples.

## Requirements

- Python 3.8+
- NVIDIA GPU with TensorRT support
- CUDA and cuDNN installed
- See `requirements.txt` for Python dependencies



## License

This project is licensed under the MIT License. See the `LICENSE` file for details.