"""Local-only MiniLM inference; no Transformers or PyTorch dependency."""
from __future__ import annotations

import json
from threading import Lock
from time import monotonic

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = (
    "onnx/model.onnx", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "config.json", "modules.json",
    "sentence_bert_config.json", "config_sentence_transformers.json",
    "1_Pooling/config.json",
)


class OnnxSentenceEncoder:
    def __init__(self, model_name: str):
        # Import here so module/API import remains lightweight.
        from src.embeddings import log_warmup, model_cache_path

        started_at = monotonic()
        if model_name != MODEL_NAME:
            raise ValueError("Unsupported embedding model.")
        path = model_cache_path(model_name)
        log_warmup("locating_cached_model", started_at, path=path, local_files_only=True)
        if not all((path / name).is_file() for name in MODEL_FILES):
            raise RuntimeError("Embedding build artifact missing; run python -m src.prefetch_model during build.")
        config = json.loads((path / "sentence_bert_config.json").read_text())
        pooling = json.loads((path / "1_Pooling/config.json").read_text())
        if (config != {"max_seq_length": 256, "do_lower_case": False}
                or pooling.get("word_embedding_dimension") != 384
                or not pooling.get("pooling_mode_mean_tokens")
                or any(value for key, value in pooling.items()
                       if key.startswith("pooling_mode_") and key != "pooling_mode_mean_tokens")):
            raise RuntimeError("Incompatible embedding artifact configuration.")
        log_warmup("importing_onnx_runtime", started_at)
        import onnxruntime as ort
        from tokenizers import Tokenizer
        log_warmup("onnx_runtime_imported", started_at)

        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=config["max_seq_length"])
        self.tokenizer.no_padding()  # The service encodes one sentence per call.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        log_warmup("constructing_onnx_session", started_at)
        self.session = ort.InferenceSession(str(path / "onnx/model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self._inference_lock = Lock()
        log_warmup("onnx_session_constructed", started_at)

    def encode(self, text: str, *, normalize_embeddings: bool = True,
               show_progress_bar: bool = False) -> np.ndarray:
        # Bound concurrent activation allocations on small instances; no cache lock.
        with self._inference_lock:
            tokens = self.tokenizer.encode(text.strip(), add_special_tokens=True)
            feeds = {"input_ids": np.asarray([tokens.ids], dtype=np.int64),
                     "attention_mask": np.asarray([tokens.attention_mask], dtype=np.int64),
                     "token_type_ids": np.asarray([tokens.type_ids], dtype=np.int64)}
            hidden = self.session.run(["last_hidden_state"], feeds)[0]
            mask = feeds["attention_mask"][..., None].astype(np.float32)
            vector = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            if normalize_embeddings:
                vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
            vector = np.asarray(vector[0], dtype=np.float32)
            if vector.shape != (384,) or not np.isfinite(vector).all():
                raise RuntimeError("Invalid embedding output.")
            return vector
