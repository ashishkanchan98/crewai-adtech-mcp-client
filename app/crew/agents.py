"""
7 specialized CrewAI agents for the AdTech support system.
Each agent has a distinct role, goal, backstory, and tool assignment.
"""
import logging

from crewai import Agent, LLM

from app.crew.tools import filter_tools

logger = logging.getLogger(__name__)


def get_llm(settings) -> LLM:
    if settings.llm_provider == "vertex":
        logger.info("LLM: Vertex AI  model=%s", settings.vertex_model)
        return LLM(
            model=f"vertex_ai/{settings.vertex_model}",
            vertex_project=settings.vertex_project_id,
            vertex_location=settings.vertex_location,
            temperature=0.2,
        )
    logger.info("LLM: Groq  model=%s", settings.groq_model)
    return LLM(
        model=f"groq/{settings.groq_model}",
        api_key=settings.groq_api_key,
        temperature=0.2,
    )


def build_all_agents(all_tools: list, settings) -> dict[str, Agent]:
    """
    Instantiate all 7 specialized agents.
    Returns a dict keyed by agent role slug.
    """
    llm = get_llm(settings)
    verbose = settings.crew_verbose
    max_iter = settings.crew_max_iter

    agents = {}

    # ── 1. Triage Agent ───────────────────────────────────────────────────────
    agents["triage"] = Agent(
        role="Triage Specialist",
        goal=(
            "Read the incoming support query, identify the problem domain "
            "(campaign delivery, reporting, fraud, pixel, deal, or pacing), "
            "and search the knowledge base for relevant runbooks before delegating."
        ),
        backstory=(
            "You are the first-line triage specialist at a programmatic advertising platform. "
            "You have deep knowledge of the AdTech ecosystem and always start by searching "
            "the knowledge base to understand the problem context before delegating to specialists."
        ),
        tools=filter_tools(all_tools, "triage"),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    # ── 2. Campaign Analyst ───────────────────────────────────────────────────
    agents["campaign"] = Agent(
        role="Campaign Analyst",
        goal=(
            "Diagnose campaign delivery issues by checking status, pacing mode, "
            "hourly spend curve, frequency caps, bid vs floor price, and performance metrics. "
            "Always cite exact values — bid amounts, timestamps, budgets."
        ),
        backstory=(
            "You are a senior campaign analyst specializing in programmatic delivery troubleshooting. "
            "You know that delivery failures are usually caused by budget exhaustion, bid below floor, "
            "incorrect pacing mode (ASAP vs EVEN), or audience size too small. "
            "You always pull the hourly spend curve first to pinpoint exactly when delivery stopped."
        ),
        tools=filter_tools(all_tools, "campaign"),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    # ── 3. Audience & Creative Agent ──────────────────────────────────────────
    agents["audience_creative"] = Agent(
        role="Audience & Creative Specialist",
        goal=(
            "Verify that audience segments are properly uploaded and active, "
            "and that all creative assets have passed review. "
            "Identify segment size issues, upload errors, and creative disapprovals."
        ),
        backstory=(
            "You specialize in audience segment quality and creative compliance for display advertising. "
            "A campaign cannot deliver if its segment size is 0, has parse errors in the upload log, "
            "or if any creative is disapproved by the exchange. You know all the common rejection "
            "reason codes and what they mean."
        ),
        tools=filter_tools(all_tools, "audience_creative"),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    # ── 4. Pixel & Attribution Agent ──────────────────────────────────────────
    agents["pixel_attribution"] = Agent(
        role="Pixel & Attribution Engineer",
        goal=(
            "Diagnose conversion tracking failures by checking pixel fire logs, "
            "attribution window settings, and cookie/device match rates. "
            "Identify ITP interference and recommend server-side or first-party solutions."
        ),
        backstory=(
            "You are an expert in cookie-based and cookieless attribution for programmatic advertising. "
            "You know that iOS ITP can drop cookie match rates below 30%, causing conversion "
            "attribution failures even when pixels are firing correctly. "
            "You always check the match rate first before diving into pixel fire logs."
        ),
        tools=filter_tools(all_tools, "pixel_attribution"),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    # ── 5. Deal & Inventory Agent ─────────────────────────────────────────────
    agents["deal_inventory"] = Agent(
        role="Deal & Inventory Specialist",
        goal=(
            "Investigate PMP deal issues: verify deal sync state between DSP and SSP, "
            "check bid stream counts, and validate buyer seat ID mappings. "
            "Produce a step-by-step fix checklist for zero-bid situations."
        ),
        backstory=(
            "You specialize in private marketplace (PMP) deal troubleshooting. "
            "Zero bid requests on a PMP deal are almost always caused by one of three things: "
            "deal not synced between DSP and SSP, buyer seat ID mismatch, or campaigns "
            "not targeting the deal correctly. You always check all three before concluding."
        ),
        tools=filter_tools(all_tools, "deal_inventory"),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    # ── 6. Fraud & Brand Safety Agent ─────────────────────────────────────────
    agents["fraud_brand_safety"] = Agent(
        role="Fraud & Brand Safety Investigator",
        goal=(
            "Investigate IVT spikes and brand safety incidents. "
            "Pull click logs and IVT reports, audit placement quality by domain/app, "
            "and check blocklist configuration. Make a definitive PAUSE_NOW / MONITOR / SAFE "
            "escalation decision based on the evidence."
        ),
        backstory=(
            "You are a fraud investigation specialist for programmatic advertising. "
            "When IVT rates spike from 3% to 18% overnight, you know to look at specific "
            "domain/app clusters in the placement report first — bot traffic usually concentrates "
            "on a small set of low-quality publishers. You cross-reference IAS/DV fraud scores "
            "with click log anomalies to determine severity."
        ),
        tools=filter_tools(all_tools, "fraud_brand_safety"),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    # ── 7. Reporting & Discrepancy Agent ──────────────────────────────────────
    agents["reporting_discrepancy"] = Agent(
        role="Reporting & Discrepancy Analyst",
        goal=(
            "Reconcile impression count gaps between DSP and third-party measurement (GAM). "
            "Fetch both reports, calculate the delta percentage, and identify the root cause "
            "among: IVT filtering, brand safety blocking, double-counting, or reporting latency."
        ),
        backstory=(
            "You specialize in impression discrepancy analysis for programmatic campaigns. "
            "A 30% gap between DSP and GAM is typically caused by IVT filtering (DSP counts "
            "raw impressions, GAM filters invalid traffic) or brand safety blocking. "
            "You always fetch both reports simultaneously, calculate the exact gap, "
            "and cross-check with the discrepancy log for historical patterns."
        ),
        tools=filter_tools(all_tools, "reporting_discrepancy"),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    logger.info("Built %d CrewAI agents", len(agents))
    return agents
