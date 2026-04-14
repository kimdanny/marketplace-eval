from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import faiss
import httpx
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

from marketplace_eval.agents.agent import Agent

load_dotenv()
logger = logging.getLogger(__name__)

INDEX_DIR = Path("index")

_RETRIEVER_REGISTRY: dict[str, type["RetrieverAgent"]] = {}


def register_retriever(key: str):
    """Class decorator that registers a RetrieverAgent subclass under *key*.

    Usage::

        @register_retriever("my_backend")
        class MyRetrieverAgent(RetrieverAgent):
            ...

    Reference in YAML config as ``backend: my_backend``.
    """

    def wrapper(cls: type["RetrieverAgent"]) -> type["RetrieverAgent"]:
        _RETRIEVER_REGISTRY[key] = cls
        return cls

    return wrapper


def _get_retriever_class(key: str) -> type["RetrieverAgent"]:
    """Look up a registered retriever class by its config key."""
    if key not in _RETRIEVER_REGISTRY:
        available = ", ".join(sorted(_RETRIEVER_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown retriever backend '{key}'. Registered types: {available}. "
            "Register a custom retriever with @register_retriever."
        )
    return _RETRIEVER_REGISTRY[key]


def _sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


@register_retriever("corpus")
class RetrieverAgent(Agent):
    def __init__(
        self,
        node_id: str,
        *,
        cost: float = 0.0,
        model_id: str | None = None,
        name: str | None = None,
        corpus_path: str | None = None,
        top_k: int = 3,
    ):
        super().__init__(node_id, cost=cost, model_id=model_id, name=name)
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.faiss_index: faiss.Index | None = None
        self.documents: List[str] = []
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Build (or load a cached) FAISS index from the corpus."""
        if self.model_id is None:
            raise ValueError("model_id is required for FAISS-based retrieval")
        if self.corpus_path is None:
            raise ValueError("corpus_path is required for FAISS-based retrieval")

        model_safe = _sanitize_model_name(self.model_id)
        corpus_stem = Path(self.corpus_path).stem
        index_subdir = INDEX_DIR / f"{model_safe}_{corpus_stem}"
        index_path = index_subdir / "index.faiss"
        docs_path = index_subdir / "documents.json"

        if index_path.exists() and docs_path.exists():
            logger.info("Loading cached FAISS index from %s", index_subdir)
            self.faiss_index = faiss.read_index(str(index_path))
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self._load_model()
            logger.info("Loaded FAISS index with %d vectors", self.faiss_index.ntotal)
            return

        logger.info(
            "Building FAISS index for %s from %s", self.model_id, self.corpus_path
        )
        self.documents = self._load_corpus(self.corpus_path)
        if not self.documents:
            raise ValueError(f"No documents found in {self.corpus_path}")

        self._load_model()

        # e5 model famaily should be encoded with the "passage: " prefix.
        is_e5 = "e5" in self.model_id.lower()
        texts = (
            [f"passage: {doc}" for doc in self.documents] if is_e5 else self.documents
        )
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        index_subdir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(index_path))
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

        logger.info(
            "Built and cached FAISS index (%d vectors) at %s",
            self.faiss_index.ntotal,
            index_subdir,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.eval()

    @staticmethod
    def _load_corpus(corpus_path: str) -> List[str]:
        documents: List[str] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    documents.append(text)
        return documents

    def _encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self._model(**encoded)
            emb = _mean_pooling(output, encoded["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def invoke(self, *, query: str, generator_id: str) -> Dict[str, Any]:
        if self.faiss_index is None:
            raise RuntimeError(
                f"FAISS index not built for {self.node_id}. Call build_index() first."
            )

        # e5 model famaily should be encoded with the "query: " prefix.
        is_e5 = self.model_id is not None and "e5" in self.model_id.lower()
        query_text = f"query: {query}" if is_e5 else query

        def _search() -> tuple[np.ndarray, np.ndarray]:
            query_emb = self._encode([query_text])
            scores, indices = self.faiss_index.search(query_emb, self.top_k)
            return scores[0], indices[0]

        scores, indices = await asyncio.to_thread(_search)

        top_documents: List[str] = []
        top_scores: List[float] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            top_documents.append(self.documents[int(idx)])
            top_scores.append(float(score))

        return {"documents": top_documents, "scores": top_scores, "query": query}

    def present(self, retrieval_result: Dict[str, Any]) -> str:
        documents = retrieval_result.get("documents", [])
        return " | ".join(documents)


class WebSearchRetrieverAgent(RetrieverAgent):
    """Base class for retrievers backed by external web search APIs."""

    def __init__(
        self,
        node_id: str,
        *,
        cost: float = 0.0,
        name: str | None = None,
        top_k: int = 3,
        timeout: float = 10.0,
    ):
        super().__init__(
            node_id,
            cost=cost,
            name=name,
            top_k=top_k,
        )
        self.timeout = timeout

    async def invoke(self, *, query: str, generator_id: str) -> Dict[str, Any]:
        documents = await self._fetch_documents(query)
        top_documents = documents[: self.top_k]
        if not top_documents:
            return {"documents": [], "scores": [], "query": query}

        # This is a simple scoring funciton.
        # Depending on web search backend, if they provide scores,
        #   override this method and change the scoring function.
        scores = [
            (len(top_documents) - rank) / len(top_documents)
            for rank in range(len(top_documents))
        ]
        return {"documents": top_documents, "scores": scores, "query": query}

    async def _fetch_documents(
        self, query: str
    ) -> List[str]:  # pragma: no cover - abstract
        raise NotImplementedError


@register_retriever("google_search")
class GoogleSearchRetrieverAgent(WebSearchRetrieverAgent):
    """
    Retriever backed by the Google Custom Search API.
    Manage your API and keys at:
    https://console.cloud.google.com/apis/api/customsearch.googleapis.com/
        Organizational google account may not be able to be used for this API.
        Limit of 10,000 queries per day for free tier.
    Documentation:
    https://developers.google.com/custom-search/v1/introduction

    Search Results fields:
    https://developers.google.com/custom-search/v1/reference/rest/v1/Search#Result

    params:
      safe: for safe search, set to "active"
      fetch_full_text: if True, fetch full text from URLs (expensive operation)
      max_text_length: max length of the text to return
      url_timeout: timeout for each URL fetch
    """

    SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        node_id: str,
        *,
        safe: str | None = "active",
        fetch_full_text: bool = True,
        max_text_length: int = 1000,
        url_timeout: float = 5.0,
        **kwargs: Any,
    ):
        if fetch_full_text:
            # lazy import of BeautifulSoup since we may not use it.
            try:
                from bs4 import BeautifulSoup  # type: ignore

                self.BeautifulSoup = BeautifulSoup
            except ModuleNotFoundError:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "BeautifulSoup is not installed. Install the 'beautifulsoup4' package to use GoogleSearchRetrieverAgent."
                )
        else:
            self.BeautifulSoup = None

        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        search_engine_id = os.getenv("GOOGLE_CSE_ID")
        if not api_key:
            raise RuntimeError("GOOGLE_SEARCH_API_KEY is required in environment.")
        if not search_engine_id:
            raise RuntimeError("GOOGLE_CSE_ID is required in environment.")
        super().__init__(node_id, **kwargs)
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.safe = safe
        self.fetch_full_text = fetch_full_text
        self.max_text_length = max_text_length
        self.url_timeout = url_timeout

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content."""
        try:
            soup = self.BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and clean it up
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Truncate if too long
            if len(text) > self.max_text_length:
                text = text[: self.max_text_length] + "..."

            return text
        except Exception:
            return ""

    async def _fetch_url_content(self, url: str) -> str:
        """Fetch content from a single URL."""
        try:
            async with httpx.AsyncClient(timeout=self.url_timeout) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                # Check if content is HTML
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    return self._extract_text_from_html(response.text)
                else:
                    # For non-HTML content, return as is (truncated)
                    text = response.text
                    if len(text) > self.max_text_length:
                        text = text[: self.max_text_length] + "..."
                    return text
        except Exception:
            return ""

    async def _fetch_documents(self, query: str) -> List[str]:
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": self.top_k,
        }
        if self.safe is not None:
            params["safe"] = self.safe

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.SEARCH_ENDPOINT, params=params)
                response.raise_for_status()
        except httpx.HTTPError:
            return []

        data = response.json()
        items = data.get("items", []) or []

        # Option 1: Only returns title and snippet (cheaper operation)
        if not self.fetch_full_text:
            documents: List[str] = []
            for item in items:
                title = item.get("title") or ""
                snippet = item.get("snippet") or ""
                combined = " ".join(part for part in [title, snippet] if part).strip()
                if combined:
                    documents.append(combined)
            return documents

        # Option 2: Fetch full text from URLs (expensive operation)
        documents: List[str] = []
        urls = [item.get("link") for item in items if item.get("link")]

        if urls:
            # Fetch all URLs concurrently
            tasks = [self._fetch_url_content(url) for url in urls]
            url_contents = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (item, content) in enumerate(zip(items, url_contents)):
                title = item.get("title") or ""
                snippet = item.get("snippet") or ""

                if isinstance(content, Exception) or not content:
                    # Fallback to snippet if URL fetch failed
                    combined = " ".join(
                        part for part in [title, snippet] if part
                    ).strip()
                else:
                    # Use full text content with title and link for context
                    combined = f"{title}\n{content}"

                if combined.strip():
                    documents.append(combined.strip())

        return documents


# Backward-compatibility alias: backend: "google" maps to GoogleSearchRetrieverAgent.
_RETRIEVER_REGISTRY["google"] = GoogleSearchRetrieverAgent


def create_retriever_agent(node_id: str, **params: Any) -> RetrieverAgent:
    """Factory helper that instantiates the correct retriever implementation.

    For corpus-backed retrievers (default), the FAISS index is built
    automatically when *corpus_path* is provided.
    """
    params = dict(params)
    backend = str(params.pop("backend", None) or "corpus").lower()

    # Preserve explicit NotImplementedError for planned-but-unbuilt backends.
    if backend in {"exa", "exa_search", "exa_research", "exa_answer"}:
        raise NotImplementedError("ExaSearchRetrieverAgent is not implemented yet")
    if backend in {"duckduckgo", "duckduckgo_search"}:
        raise NotImplementedError(
            "DuckDuckGoSearchRetrieverAgent is not implemented yet"
        )

    cls = _get_retriever_class(backend)
    agent = cls(node_id, **params)

    # Post-construction: build FAISS index for corpus-backed retrievers.
    if isinstance(agent, RetrieverAgent) and not isinstance(
        agent, WebSearchRetrieverAgent
    ):
        if agent.corpus_path is not None:
            agent.build_index()

    return agent
