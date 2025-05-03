import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Inference")


class HostDeviceMem:
    """Pair of host and device memory for TensorRT inference."""
    def __init__(self, host_mem, device_mem, binding_name, shape):
        self.host = host_mem
        self.device = device_mem
        self.binding_name = binding_name
        self.shape = shape


class TRTInference:
    """Manages TensorRT engine inference with dynamic shapes."""
    def __init__(self, engine_path, max_batch_size=32):
        """
        Initialize the TRTInference class.

        Args:
            engine_path (str): Path to the TensorRT engine file.
            max_batch_size (int): Maximum batch size supported by the engine.

        Raises:
            RuntimeError: If engine deserialization fails.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.max_batch_size = max_batch_size

        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Initialize buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # Check if engine has dynamic shapes
        self.dynamic_shapes = False
        for tensor_name in self.engine:
            shape = self.engine.get_tensor_shape(tensor_name)
            if shape[0] == -1:
                self.dynamic_shapes = True
                break

    def allocate_buffers(self, batch_size):
        """Allocate host and device buffers for the given batch size."""
        self.inputs.clear()
        self.outputs.clear()
        self.bindings.clear()

        for tensor_name in self.engine:
            binding_shape = list(self.engine.get_tensor_shape(tensor_name))
            
            # Replace dynamic dimension (-1) with batch_size for inputs
            if self.dynamic_shapes and binding_shape[0] == -1:
                binding_shape[0] = batch_size
            
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            
            # Store input/output buffers
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(HostDeviceMem(host_mem, device_mem, tensor_name, binding_shape))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem, tensor_name, binding_shape))

        # Set shape for dynamic inputs
        if self.dynamic_shapes:
            for input_mem in self.inputs:
                self.context.set_input_shape(input_mem.binding_name, tuple(input_mem.shape))

    def infer(self, input_data):
        """
        Perform inference with the provided input data.

        Args:
            input_data (np.ndarray): Input array with shape (batch_size, channels, height, width).

        Returns:
            list: List of output numpy arrays.

        Raises:
            ValueError: If batch size exceeds maximum.
        """
        batch_size = input_data.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")

        # Allocate buffers for the current batch size
        self.allocate_buffers(batch_size)

        # Copy input data to host memory
        for input_mem in self.inputs:
            np.copyto(input_mem.host, input_data.ravel())

        # Transfer inputs to device
        for input_mem in self.inputs:
            cuda.memcpy_htod_async(input_mem.device, input_mem.host, self.stream)

        # Execute inference
        self.context.execute_v2(bindings=self.bindings)

        # Transfer outputs back to host
        for output_mem in self.outputs:
            cuda.memcpy_dtoh_async(output_mem.host, output_mem.device, self.stream)

        # Synchronize stream
        self.stream.synchronize()

        # Return output as reshaped numpy array
        return [output_mem.host.reshape(output_mem.shape) for output_mem in self.outputs]

    def __del__(self):
        """Clean up CUDA resources."""
        for mem in self.inputs + self.outputs:
            mem.device.free()