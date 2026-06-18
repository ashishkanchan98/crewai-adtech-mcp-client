import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.config import settings
from app.db.database import init_db


# ── Logging ───────────────────────────────────────────────────────────────────
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


if settings.env == "prod":
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logging.root.setLevel(logging.INFO)
    logging.root.handlers = [handler]
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

logger = logging.getLogger(__name__)


# ── API key middleware ────────────────────────────────────────────────────────
class _ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if settings.api_key and request.url.path.startswith("/api/"):
            key = request.headers.get("X-API-Key", "")
            if key != settings.api_key:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database initialized  url=%s", settings.database_url)

    from app.crew.tools import connect_mcp_adapter
    from app.crew.crews import CrewRouter

    mcp_adapter = None
    mcp_tools = []

    try:
        mcp_adapter, mcp_tools = connect_mcp_adapter(settings.mcp_server_url)
        logger.info(
            "Connected to real-python-adtech-mcp-server  url=%s  tools=%d",
            settings.mcp_server_url, len(mcp_tools),
        )
    except Exception as exc:
        logger.warning(
            "MCP server unreachable at startup (%s). "
            "Agents will run with KB-only tools until server is available.",
            exc,
        )

    crew_router = CrewRouter(mcp_tools, settings)
    app.state.crew_router = crew_router
    app.state.mcp_tools_count = len(mcp_tools)

    logger.info(
        "crewai-adtech-mcp-client ready  env=%s  llm=%s  mcp_tools=%d  port=%s",
        settings.env, settings.llm_provider, len(mcp_tools),
        os.environ.get("PORT", str(settings.port)),
    )

    yield

    if mcp_adapter is not None:
        try:
            mcp_adapter.__exit__(None, None, None)
            logger.info("MCPServerAdapter connection closed")
        except Exception:
            pass

    logger.info("crewai-adtech-mcp-client shut down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AdTech CrewAI Client",
    description=(
        "Multi-agent AdTech support system built with CrewAI. "
        "7 specialized agents routed across 7 use cases via MCPServerAdapter."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.env != "prod" else None,
    redoc_url="/redoc" if settings.env != "prod" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.api_key:
    app.add_middleware(_ApiKeyMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)
