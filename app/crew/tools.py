"""
Tool management for crewai-adtech-mcp-client.

MCP tools come from real-python-adtech-mcp-server via MCPServerAdapter (SSE transport).
KB search tool (ChromaDB local / Vertex AI Search GCP) is created natively.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Tool name groups per agent ─────────────────────────────────────────────────
AGENT_TOOL_NAMES: dict[str, list[str]] = {
    "triage":               ["searchKnowledgeBase"],
    "campaign":             ["getCampaignStatus", "getPacingSettings",
                             "getCampaignFrequencySettings", "getHourlySpendCurve",
                             "getPerformanceMetrics"],
    "audience_creative":    ["getSegmentStatus", "getSegmentUploadLog",
                             "getCreativeStatus", "getCreativeAsset"],
    "pixel_attribution":    ["getPixelFireLog", "getAttributionWindowSettings",
                             "getConversionMatchRate"],
    "deal_inventory":       ["getDealStatus", "getDealBidStream", "getSeatMapping"],
    "fraud_brand_safety":   ["getClickLog", "getIVTReport",
                             "getBrandSafetySettings", "getPlacementReport"],
    "reporting_discrepancy":["getDSPImpressionReport", "getGAMReport", "getDiscrepancyLog"],
}


def filter_tools(all_tools: list, agent_key: str) -> list:
    """Return only the tools assigned to the given agent."""
    names = set(AGENT_TOOL_NAMES.get(agent_key, []))
    return [t for t in all_tools if t.name in names]


# ── MCPServerAdapter connection ────────────────────────────────────────────────

def connect_mcp_adapter(mcp_server_url: str):
    """
    Open MCPServerAdapter context to the real-python-adtech-mcp-server (SSE).
    Returns (adapter, tools_list). Caller owns the adapter context — must call
    adapter.__exit__(None, None, None) at shutdown.
    """
    from crewai_tools import MCPServerAdapter

    server_params = {"url": mcp_server_url, "transport": "sse"}
    adapter = MCPServerAdapter(server_params)
    tools = adapter.__enter__()  # opens SSE connection, discovers tools
    logger.info("MCPServerAdapter connected  url=%s  tools=%d", mcp_server_url, len(tools))
    return adapter, tools


# ── Knowledge Base tool ────────────────────────────────────────────────────────

_chroma_collection = None


def _get_chroma_collection(settings):
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    collection = client.get_or_create_collection("adtech_kb", embedding_function=ef)

    if collection.count() == 0:
        _seed_chroma(collection, settings.kb_docs_dir)

    logger.info("ChromaDB ready  docs=%d  path=%s", collection.count(), settings.chroma_db_path)
    _chroma_collection = collection
    return collection


def _seed_chroma(collection, kb_docs_dir: str) -> None:
    docs_path = Path(kb_docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"kb-docs not found at '{kb_docs_dir}'")

    json_files = sorted(docs_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {kb_docs_dir}")

    ids, documents, metadatas = [], [], []
    for f in json_files:
        with open(f) as fp:
            doc = json.load(fp)
        struct = doc["structData"]
        title = struct.get("title", "")
        content = struct.get("content", "")
        ids.append(doc["id"])
        documents.append(f"{title}\n\n{content}")
        metadatas.append({
            "title": title,
            "category": struct.get("category", ""),
            "link": struct.get("link", ""),
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("ChromaDB seeded with %d documents from %s", len(ids), kb_docs_dir)


def _kb_search_local(query: str, max_results: int, settings) -> dict:
    try:
        collection = _get_chroma_collection(settings)
        n = min(max_results, collection.count())
        results = collection.query(query_texts=[query], n_results=n)
        docs = []
        for doc_id, meta, distance in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            docs.append({
                "id": doc_id,
                "title": meta.get("title", ""),
                "snippet": meta.get("category", ""),
                "link": meta.get("link", ""),
                "score": round(1 - distance, 3),
            })
        return {"query": query, "count": len(docs), "results": docs, "source": "local"}
    except Exception as exc:
        logger.error("ChromaDB search failed: %s", exc)
        return {"error": str(exc), "query": query, "count": 0, "results": []}


def _kb_search_gcp(query: str, max_results: int, settings) -> dict:
    if not settings.gcp_project_id:
        return {"error": "GCP_PROJECT_ID required when KB_PROVIDER=gcp", "count": 0, "results": []}
    try:
        from google.cloud import discoveryengine_v1 as discoveryengine
        client = discoveryengine.SearchServiceClient()
        serving_config = client.serving_config_path(
            project=settings.gcp_project_id,
            location="global",
            data_store=settings.kb_datastore_id,
            serving_config=settings.kb_serving_config,
        )
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=max_results,
        )
        response = client.search(request)
        results = [{"id": r.id, "title": "", "snippet": "", "link": ""} for r in response.results]
        return {"query": query, "count": len(results), "results": results, "source": "gcp"}
    except Exception as exc:
        logger.error("Vertex AI Search failed: %s", exc)
        return {"error": str(exc), "count": 0, "results": []}


class _KbInput(BaseModel):
    query: str = Field(description="Natural language search query")
    max_results: int = Field(default=3, description="Max docs to return")


class KBSearchTool(BaseTool):
    name: str = "searchKnowledgeBase"
    description: str = (
        "Searches the AdTech support knowledge base for troubleshooting guides, "
        "policy documents, and how-to articles. Use for runbook lookups and best practices."
    )
    args_schema: Type[BaseModel] = _KbInput
    _settings: Any = None

    def __init__(self, settings):
        super().__init__()
        self._settings = settings

    def _run(self, query: str, max_results: int = 3) -> str:
        if self._settings.kb_provider == "gcp":
            result = _kb_search_gcp(query, max_results, self._settings)
        else:
            result = _kb_search_local(query, max_results, self._settings)
        return json.dumps(result)


def create_kb_tool(settings) -> KBSearchTool:
    return KBSearchTool(settings=settings)
