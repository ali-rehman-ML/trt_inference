import os
import logging
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    Optimized for maximum resource usage.
    """

    def __init__(self, verbose=False, workspace=16):
        """
        Initialize the EngineBuilder.

        Args:
            verbose (bool): If True, set a higher verbosity level for the TensorRT logger.
            workspace (int): Max memory workspace to allow, in GB.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Set workspace memory pool size
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            onnx_path (str): Path to the ONNX graph to load.

        Raises:
            SystemExit: If parsing the ONNX file fails.
        """
        self.network = self.builder.create_network(1)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error(f"Failed to load ONNX file: {onnx_path}")
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}"
            )
        for output in outputs:
            log.info(
                f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}"
            )
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

    def create_engine(self, engine_path, precision="fp16", use_int8=False):
        """
        Build the TensorRT engine and serialize it to disk.

        Args:
            engine_path (str): Path where to serialize the engine.
            precision (str): Datatype for the engine ('fp32', 'fp16').
            use_int8 (bool): Enable INT8 precision mode if hardware supports it.

        Raises:
            RuntimeError: If engine creation fails.
        """
        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        log.info(f"Building {precision} Engine in {engine_path}")

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
            raise RuntimeError("Failed to create engine")

        with open(engine_path, "wb") as f:
            log.info(f"Serializing engine to file: {engine_path}")
            f.write(engine_bytes)