import json
from typing import List, Optional, Union

import requests


class LocalHttpEmbeddingModel:
    """
    Simple embedding client for a custom HTTP endpoint that accepts
    a JSON body with key "inputs": [text, ...] and returns embeddings.

    Expected response formats (any of these will be parsed):
    - {"embeddings": [[...], [...]]}
    - {"data": {"embeddings": [[...], [...]]}}
    - [[...], [...]]  # bare list of vectors
    - {"vectors": [[...], [...]]}
    - {"data": [{"embedding": [...]}, {"embedding": [...]}]}
    """

    def __init__(self, llm_kwargs: Optional[dict] = None):
        self.llm_kwargs = llm_kwargs or {}

    def _get_endpoint(self, llm_kwargs: Optional[dict] = None) -> str:
        if llm_kwargs is None:
            llm_kwargs = self.llm_kwargs
        if llm_kwargs is None:
            raise RuntimeError("llm_kwargs is not provided!")
        # bridge_all_embed provides embed_endpoint via model info
        endpoint = llm_kwargs.get("embed_endpoint") or self.llm_kwargs.get("embed_endpoint")
        if not endpoint:
            raise RuntimeError("embed_endpoint is missing in llm_kwargs for LocalHttpEmbeddingModel")
        return endpoint

    def get_query_embedding(self, query: str):
        return self.compute_embedding(query)

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False):
        return self.compute_embedding(texts, batch_mode=True)

    def compute_embedding(
        self,
        text: Union[str, List[str]] = "这是要计算嵌入的文本",
        llm_kwargs: Optional[dict] = None,
        batch_mode: bool = False,
    ):
        endpoint = self._get_endpoint(llm_kwargs)

        if batch_mode:
            assert isinstance(text, list), "Batch mode requires a list of strings"
            inputs = text
        else:
            assert isinstance(text, str), "Single mode requires a string"
            inputs = [text]

        headers = {"Content-Type": "application/json"}
        # Optional authorization header support
        api_key = None
        if llm_kwargs is not None:
            api_key = llm_kwargs.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {"inputs": inputs}

        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()

        embeddings = self._extract_embeddings(data)
        if embeddings is None:
            raise RuntimeError(f"Unexpected embedding response format from {endpoint}: {data}")

        if batch_mode:
            return embeddings
        else:
            return embeddings[0]

    def _extract_embeddings(self, data):
        # Common patterns
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data
        if isinstance(data, dict):
            if "embeddings" in data and isinstance(data["embeddings"], list):
                return data["embeddings"]
            if "vectors" in data and isinstance(data["vectors"], list):
                return data["vectors"]
            if "data" in data:
                inner = data["data"]
                if isinstance(inner, list) and inner and isinstance(inner[0], dict) and "embedding" in inner[0]:
                    return [d.get("embedding") for d in inner]
                if isinstance(inner, dict):
                    if "embeddings" in inner and isinstance(inner["embeddings"], list):
                        return inner["embeddings"]
        return None


