import os
import sys
import logging
import argparse
import numpy as np
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    Optimized for maximum resource usage and supports dynamic shapes.
    """

    def __init__(self, verbose=False, workspace=16):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in GB. Increased for maximum optimization.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()

        # Increase the workspace memory pool size for maximum performance
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.network = None
        self.parser = None
        self.profile = None
        self.dynamic_shapes = False

    def create_network(self, onnx_path, min_batch=1, opt_batch=16, max_batch=32):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        Supports dynamic shapes with optimization profiles.
        :param onnx_path: The path to the ONNX graph to load.
        :param min_batch: Minimum batch size for dynamic shapes.
        :param opt_batch: Optimal batch size for dynamic shapes.
        :param max_batch: Maximum batch size for dynamic shapes.
        """
        # Create network with explicit batch flag
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        # Check for dynamic shapes (any dimension with -1)
        self.dynamic_shapes = any(
            any(dim == -1 for dim in input.shape) for input in inputs
        )

        log.info("Network Description")
        for input in inputs:
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )

        if self.dynamic_shapes:
            log.info("Dynamic shapes detected, creating optimization profile")
            self.profile = self.builder.create_optimization_profile()
            for input in inputs:
                input_shape = input.shape
                min_shape = [min_batch if dim == -1 else dim for dim in input_shape]
                opt_shape = [opt_batch if dim == -1 else dim for dim in input_shape]
                max_shape = [max_batch if dim == -1 else dim for dim in input_shape]

                self.profile.set_shape(
                    input.name,
                    min_shape,
                    opt_shape,
                    max_shape
                )
            self.config.add_optimization_profile(self.profile)
        else:
            # For static shapes, use the first dimension as batch size
            for input in inputs:
                if input.shape[0] > 0:
                    self.builder.max_batch_size = input.shape[0]
                    break

    def create_engine(self, engine_path, precision="fp16", use_int8=False):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16'.
        :param use_int8: Enable INT8 precision mode if hardware supports it.
        """
        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        # Set precision flags
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)

        if use_int8:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)

        # Build and serialize the engine
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            sys.exit(1)

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)
