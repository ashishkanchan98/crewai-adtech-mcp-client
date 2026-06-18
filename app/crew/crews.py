"""
7 CrewAI crew builders — one per use case.
CrewRouter dispatches incoming queries to the right crew.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from crewai import Crew, Process

from app.crew.agents import build_all_agents
from app.crew.router import ALL_USE_CASES, classify_query, get_use_case_label
from app.crew.tasks import (
    uc1_campaign_delivery_tasks,
    uc2_reporting_discrepancy_tasks,
    uc3_ivt_fraud_tasks,
    uc4_pmp_deal_tasks,
    uc5_pixel_attribution_tasks,
    uc6_pre_launch_audit_tasks,
    uc7_budget_pacing_tasks,
)

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)

# Map use case keys to task factory functions
_TASK_FACTORIES = {
    "delivery_failure":      uc1_campaign_delivery_tasks,
    "reporting_discrepancy": uc2_reporting_discrepancy_tasks,
    "ivt_fraud":             uc3_ivt_fraud_tasks,
    "pmp_deal":              uc4_pmp_deal_tasks,
    "pixel_attribution":     uc5_pixel_attribution_tasks,
    "pre_launch_audit":      uc6_pre_launch_audit_tasks,
    "budget_pacing":         uc7_budget_pacing_tasks,
}

# Agents participating in each use case
_USE_CASE_AGENTS = {
    "delivery_failure":      ["triage", "campaign", "audience_creative", "deal_inventory"],
    "reporting_discrepancy": ["triage", "reporting_discrepancy", "fraud_brand_safety"],
    "ivt_fraud":             ["triage", "fraud_brand_safety", "campaign"],
    "pmp_deal":              ["triage", "deal_inventory", "campaign"],
    "pixel_attribution":     ["triage", "pixel_attribution", "campaign"],
    "pre_launch_audit":      ["triage", "campaign", "audience_creative", "pixel_attribution",
                              "deal_inventory", "fraud_brand_safety"],
    "budget_pacing":         ["campaign", "audience_creative"],
}


@dataclass
class CrewRunResult:
    use_case: str
    use_case_label: str
    agents_used: list[str]
    final_answer: str
    task_outputs: list[dict]   # [{agent, role, output}]


class CrewRouter:
    """
    Holds the shared agent pool and dispatches queries to the correct crew.
    All 7 agents are instantiated once; crews are assembled per-request
    with only the relevant agents and their tasks.
    """

    def __init__(self, mcp_tools: list, settings):
        from app.crew.tools import create_kb_tool
        kb_tool = create_kb_tool(settings)
        all_tools = list(mcp_tools) + [kb_tool]

        self._agents = build_all_agents(all_tools, settings)
        self._settings = settings
        logger.info(
            "CrewRouter initialized  agents=%d  mcp_tools=%d",
            len(self._agents), len(mcp_tools),
        )

    def _build_crew(self, use_case: str, query_ctx: dict) -> Crew:
        task_factory = _TASK_FACTORIES[use_case]
        tasks = task_factory(self._agents, query_ctx)

        agent_keys = _USE_CASE_AGENTS[use_case]
        agents = [self._agents[k] for k in agent_keys if k in self._agents]

        process = Process.sequential

        return Crew(
            agents=agents,
            tasks=tasks,
            process=process,
            verbose=self._settings.crew_verbose,
        )

    def _run_crew_sync(self, use_case: str, query_ctx: dict) -> CrewRunResult:
        """Synchronous execution — runs in a thread pool from async context."""
        crew = self._build_crew(use_case, query_ctx)
        use_case_label = get_use_case_label(use_case)
        agent_keys = _USE_CASE_AGENTS.get(use_case, [])

        logger.info("Running crew '%s'  agents=%s", use_case_label, agent_keys)

        result = crew.kickoff(inputs=query_ctx)
        final_answer = str(result) if result else "Crew completed but produced no output."

        # Collect per-task outputs
        task_outputs = []
        for task in crew.tasks:
            agent_role = task.agent.role if task.agent else "Unknown"
            output = task.output.raw if (task.output and hasattr(task.output, "raw")) else ""
            task_outputs.append({
                "agent": task.agent.role if task.agent else "Unknown",
                "role": agent_role,
                "output": output,
            })

        logger.info("Crew '%s' completed  tasks=%d", use_case_label, len(task_outputs))

        return CrewRunResult(
            use_case=use_case,
            use_case_label=use_case_label,
            agents_used=[self._agents[k].role for k in agent_keys if k in self._agents],
            final_answer=final_answer,
            task_outputs=task_outputs,
        )

    async def run(self, query: str, query_ctx: dict) -> CrewRunResult:
        """Classify the query and run the matching crew asynchronously."""
        use_case = classify_query(query)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            self._run_crew_sync,
            use_case,
            query_ctx,
        )
        return result

    @property
    def available_crews(self) -> list[str]:
        return ALL_USE_CASES
