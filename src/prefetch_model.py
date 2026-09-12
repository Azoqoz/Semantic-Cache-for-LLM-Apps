"""Build-time preparation: python -m src.prefetch_model (never imported by the API)."""
from src.config import settings
from src.embeddings import SentenceTransformer, model_cache_path


def prefetch() -> None:
    from sentence_transformers import SentenceTransformer as Model

    path = model_cache_path(settings.embedding_model)
    if not path.exists():
        # Use the configured upstream model, preserving its modules and weights.
        model = Model(settings.embedding_model)
        model.save_pretrained(str(path))
        del model
    # Validate the exact runtime loader offline; an incomplete artifact fails build.
    SentenceTransformer(settings.embedding_model)
    print(f"Prepared {settings.embedding_model} at {path}")


if __name__ == "__main__":
    prefetch()
