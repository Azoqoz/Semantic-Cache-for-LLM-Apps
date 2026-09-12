"""Build-time preparation: python -m src.prefetch_model (never imported by the API)."""
from src.config import settings
from src.embeddings import model_cache_path
from src.onnx_embeddings import MODEL_NAME, MODEL_REVISION, MODEL_FILES, OnnxSentenceEncoder


def prefetch() -> None:
    from huggingface_hub import snapshot_download

    if settings.embedding_model != MODEL_NAME:
        raise ValueError("Unsupported embedding model.")
    path = model_cache_path(settings.embedding_model)
    snapshot_download(repo_id=settings.embedding_model, revision=MODEL_REVISION,
                      local_dir=str(path), allow_patterns=list(MODEL_FILES))
    # Exercise the same local-only session, tokenizer, pooling and normalization.
    # Any missing/corrupt file or invalid inference output fails the build.
    OnnxSentenceEncoder(settings.embedding_model).encode("Semantic cache warm-up.")
    print(f"Prepared {settings.embedding_model} at {path}")


if __name__ == "__main__":
    prefetch()
