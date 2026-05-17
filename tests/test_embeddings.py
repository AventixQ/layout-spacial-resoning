from layout_spatial_reasoning.embeddings.provider import embed_texts
from layout_spatial_reasoning.embeddings.provider import embedding_function_from_env
from layout_spatial_reasoning.embeddings.similarity import cosine_similarity


def test_local_embeddings_are_deterministic_and_similar_for_same_text():
    first, second = embed_texts(["Postal code", "Postal code"])

    assert first == second
    assert cosine_similarity(first, second) == 1.0


def test_embedding_function_from_env_defaults_to_local(monkeypatch):
    monkeypatch.setenv("DEFAULT_EMBEDDING_PROVIDER", "local")

    assert embedding_function_from_env() is embed_texts
