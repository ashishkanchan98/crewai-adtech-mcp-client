crewai-adtech-mcp-client
========================
Multi-agent AdTech support system built with CrewAI.
Connects to real-python-adtech-mcp-server via MCPServerAdapter (SSE transport).

7 Specialized Agents
---------------------
1. Triage Agent              — classifies query, searches KB
2. Campaign Analyst          — status, pacing, spend, bid vs floor
3. Audience & Creative       — segment readiness, creative compliance
4. Pixel & Attribution       — conversion tracking, ITP, match rate
5. Deal & Inventory          — PMP sync, bid stream, seat IDs
6. Fraud & Brand Safety      — IVT, placement quality, blocklists
7. Reporting & Discrepancy   — DSP vs GAM gap analysis

7 Use Cases / Crews
--------------------
1. Campaign Not Delivering        → Triage + Campaign + Audience + Deal
2. Reporting Discrepancy          → Reporting + Fraud/BS + KB → Reconciliation
3. IVT / Fraud Spike              → Fraud → Placement + BS + Campaign → Escalation
4. PMP Deal Zero Bids             → Deal → Campaign + Seat + KB → Fix Checklist
5. Pixel Attribution Drop         → Pixel → Match Rate + KB + Campaign → Mitigation
6. Pre-Launch Audit (parallel)    → All 6 agents fire simultaneously → GO/NO-GO
7. Budget Pacing Optimization     → Campaign → Audience + Performance → Recommendations

Quick Start
-----------
1. Start real-python-adtech-mcp-server (SSE on port 8085):
   cd ../real-python-adtech-mcp-server
   python server.py

2. Set GROQ_API_KEY in .env

3. (Optional) Copy kb-docs from python-adtech-mcp-client:
   cp -r ../python-adtech-mcp-client/kb-docs ./kb-docs

4. Install dependencies:
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt

5. Run the client:
   uvicorn app.main:app --reload --port 8090

6. Open http://localhost:8090

API Endpoints
--------------
POST /api/v1/crew/query          — Submit query, auto-routes to correct crew
GET  /api/v1/crew/tickets        — List all tickets
GET  /api/v1/crew/tickets/{id}   — Get ticket by ID
GET  /api/v1/crew/health-check   — Health + tool count

Ports
------
real-python-adtech-mcp-server  :8085  (SSE MCP server)
crewai-adtech-mcp-client       :8090  (this service)
python-adtech-mcp-client       :8080  (LangGraph version)
python-adtech-mcp-server       :8082  (REST MCP server)
