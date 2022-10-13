import onnxruntime as ort
import torch
import numpy as np


Gb = int


class OnnxModelAdapter:
    def __init__(self, model_path: str, device_id: int = 0, gpu_mem_limit: Gb = 0) -> None:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': gpu_mem_limit * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.ort_input_name = self.session.get_inputs()[0].name

    # Using __call__ to match torch models inference interface
    def __call__(self, image_batch: torch.Tensor) -> torch.Tensor:
        np_image_batch: np.ndarray = image_batch.detach().cpu().numpy().astype(np.float32)
        ort_input = {self.ort_input_name: np_image_batch}
        embedding_batch, *__ = self.session.run(None, ort_input)

        return torch.from_numpy(embedding_batch)
