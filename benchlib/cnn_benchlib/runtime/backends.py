from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np


@dataclass
class CnnRuntimeResult:
    output: np.ndarray
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.infer_ms + self.postprocess_ms


class FloatCnnRuntime:
    def __init__(self, model_path: str):
        import tensorflow as tf
        self.tf = tf
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def run(self, image: np.ndarray) -> CnnRuntimeResult:
        t0 = time.perf_counter_ns()
        batch = np.expand_dims(image.astype(np.float32), axis=0)
        t1 = time.perf_counter_ns()
        outputs = self.model(batch, training=False)
        t2 = time.perf_counter_ns()
        out = np.asarray(outputs.numpy())
        t3 = time.perf_counter_ns()
        return CnnRuntimeResult(out, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6)


class TfliteCnnRuntime:
    def __init__(self, model_path: str, num_threads: int = 1):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.interpreter.allocate_tensors()

    def run(self, image: np.ndarray) -> CnnRuntimeResult:
        t0 = time.perf_counter_ns()
        batch = np.expand_dims(image.astype(np.float32), axis=0)
        t1 = time.perf_counter_ns()
        self.interpreter.set_tensor(self.input_index, batch)
        self.interpreter.invoke()
        t2 = time.perf_counter_ns()
        out = np.asarray(self.interpreter.get_tensor(self.output_index))
        t3 = time.perf_counter_ns()
        return CnnRuntimeResult(out, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6)


class XmodelCnnRuntime:
    def __init__(self, xmodel_path: str):
        try:
            import xir  # type: ignore
            import vart  # type: ignore
        except Exception as exc:
            raise RuntimeError("No se pudo importar xir/vart. Este runtime requiere entorno Vitis AI en la ZCU102.") from exc
        graph = xir.Graph.deserialize(xmodel_path)
        root = graph.get_root_subgraph()
        subgraphs = [sg for sg in root.toposort_child_subgraph() if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"]
        if not subgraphs:
            raise RuntimeError(f"No se encontraron subgrafos DPU en {xmodel_path}")
        self.runner = vart.Runner.create_runner(subgraphs[0], "run")
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()

    def run(self, image: np.ndarray) -> CnnRuntimeResult:
        t0 = time.perf_counter_ns()
        batch = np.expand_dims(image.astype(np.float32), axis=0)
        t1 = time.perf_counter_ns()
        input_data = np.empty(tuple(self.input_tensors[0].dims), dtype=np.float32)
        output_data = np.empty(tuple(self.output_tensors[0].dims), dtype=np.float32)
        np.copyto(input_data, batch)
        job_id = self.runner.execute_async([input_data], [output_data])
        self.runner.wait(job_id)
        t2 = time.perf_counter_ns()
        out = np.asarray(output_data)
        t3 = time.perf_counter_ns()
        return CnnRuntimeResult(out, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6)
