"""
Task factory functions for each of the 7 AdTech use cases.
Each function receives the agent dict and a query context dict, returns a list[Task].
"""
from crewai import Task


def _ctx_str(ctx: dict) -> str:
    parts = []
    for k, v in ctx.items():
        if v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts) if parts else "No additional context provided."


# ── Use Case 1: Campaign Not Delivering ───────────────────────────────────────
def uc1_campaign_delivery_tasks(agents: dict, ctx: dict) -> list[Task]:
    context_str = _ctx_str(ctx)

    triage = Task(
        description=(
            f"A campaign is not delivering. Classify the problem domain and search the "
            f"knowledge base for the delivery troubleshooting runbook.\n\nQuery context:\n{context_str}"
        ),
        expected_output=(
            "Domain classification (budget / bid / creative / segment / deal issue) "
            "and relevant KB doc snippet."
        ),
        agent=agents["triage"],
    )

    campaign_check = Task(
        description=(
            f"Investigate the campaign health for {ctx.get('campaign_id', 'the campaign')}. "
            f"Check: status (active/paused), budget remaining, bid vs floor price, pacing mode, "
            f"hourly spend curve (last 3 days), and frequency settings.\n\nContext:\n{context_str}"
        ),
        expected_output=(
            "List of campaign-level issues with exact values: bid amount, floor price, "
            "budget hit timestamp, pacing mode, frequency cap status."
        ),
        agent=agents["campaign"],
        context=[triage],
    )

    audience_check = Task(
        description=(
            f"Check segment readiness and creative compliance for {ctx.get('campaign_id', 'the campaign')}. "
            f"Verify: segment size > 0, no upload parse errors, all creatives approved.\n\nContext:\n{context_str}"
        ),
        expected_output=(
            "Segment status (size, match rate), upload log errors if any, "
            "creative review status for each creative."
        ),
        agent=agents["audience_creative"],
        context=[triage],
    )

    deal_check = Task(
        description=(
            f"If this campaign uses PMP deals, check deal sync state and bid stream for "
            f"{ctx.get('deal_id', 'any associated deals')}.\n\nContext:\n{context_str}"
        ),
        expected_output=(
            "Deal sync status, bid request count (last 6h), seat ID mapping check. "
            "If no deal — confirm open auction and skip."
        ),
        agent=agents["deal_inventory"],
        context=[triage],
    )

    synthesis = Task(
        description=(
            "Synthesize all findings from the campaign, audience, and deal checks. "
            "Rank issues by severity and produce a prioritized action plan with exact fix steps."
        ),
        expected_output=(
            "Prioritized list of issues (ranked 1-N by severity) with exact fix steps for each. "
            "Include a one-line resolution summary at the top."
        ),
        agent=agents["triage"],
        context=[triage, campaign_check, audience_check, deal_check],
    )

    return [triage, campaign_check, audience_check, deal_check, synthesis]


# ── Use Case 2: Reporting Discrepancy ─────────────────────────────────────────
def uc2_reporting_discrepancy_tasks(agents: dict, ctx: dict) -> list[Task]:
    context_str = _ctx_str(ctx)

    triage = Task(
        description=(
            f"There is a DSP vs GAM impression discrepancy. Search the KB for the "
            f"discrepancy investigation runbook.\n\nContext:\n{context_str}"
        ),
        expected_output="Relevant KB runbook for DSP/GAM discrepancy investigation.",
        agent=agents["triage"],
    )

    fetch_reports = Task(
        description=(
            f"Fetch DSP impression report and GAM report for {ctx.get('campaign_id', 'the campaign')} "
            f"and check the historical discrepancy log. Calculate the exact gap percentage."
        ),
        expected_output=(
            "DSP impression count, GAM impression count, gap percentage, "
            "and historical discrepancy trend from the log."
        ),
        agent=agents["reporting_discrepancy"],
        context=[triage],
    )

    fraud_check = Task(
        description=(
            f"Investigate IVT filtering as a cause of the discrepancy for "
            f"{ctx.get('campaign_id', 'the campaign')}. "
            f"Check IVT report and click logs for anomalies."
        ),
        expected_output=(
            "IVT rate (SIVT + GIVT breakdown), domains with highest invalid traffic rates. "
            "Estimated impressions filtered by IVT vendor."
        ),
        agent=agents["fraud_brand_safety"],
        context=[fetch_reports],
    )

    brand_safety_check = Task(
        description=(
            f"Check if brand safety blocking is reducing GAM counts for "
            f"{ctx.get('line_item_id', 'the line item')}. "
            f"Review brand safety settings and placement report."
        ),
        expected_output=(
            "Brand safety tier, blocked domains count, estimated impressions lost "
            "to brand safety filtering."
        ),
        agent=agents["fraud_brand_safety"],
        context=[fetch_reports],
    )

    reconciliation = Task(
        description=(
            "Reconcile the discrepancy delta using all findings. Identify the primary root cause "
            "among: IVT filtering, brand safety blocking, double-counting, or reporting latency."
        ),
        expected_output=(
            "Root cause with percentage attribution (e.g. '22% gap = 18% IVT filtering + 4% brand safety'). "
            "Recommended fix steps."
        ),
        agent=agents["reporting_discrepancy"],
        context=[triage, fetch_reports, fraud_check, brand_safety_check],
    )

    return [triage, fetch_reports, fraud_check, brand_safety_check, reconciliation]


# ── Use Case 3: IVT / Fraud Spike ────────────────────────────────────────────
def uc3_ivt_fraud_tasks(agents: dict, ctx: dict) -> list[Task]:
    context_str = _ctx_str(ctx)

    fraud_pull = Task(
        description=(
            f"An IVT spike has been detected. Pull click logs and IVT report for "
            f"{ctx.get('campaign_id', 'the advertiser')}. "
            f"Identify the magnitude and timing of the spike.\n\nContext:\n{context_str}"
        ),
        expected_output=(
            "IVT rate before/after spike, SIVT vs GIVT breakdown, "
            "click log anomalies (IP clusters, suspicious user agents, device ID patterns)."
        ),
        agent=agents["fraud_brand_safety"],
    )

    placement_audit = Task(
        description=(
            f"Identify which domains and apps are responsible for the IVT spike "
            f"in the placement report for the affected campaigns. "
            f"Flag any domains with IVT rate > 10%."
        ),
        expected_output=(
            "Top 5 domains/apps by IVT rate with impression counts. "
            "Recommendation: add to blocklist or pause publisher."
        ),
        agent=agents["fraud_brand_safety"],
        context=[fraud_pull],
    )

    brand_safety_check = Task(
        description=(
            f"Check if brand safety vendor (IAS/DV) blocklists are current and active "
            f"for {ctx.get('line_item_id', 'affected line items')}. "
            f"Verify brand safety settings are properly configured."
        ),
        expected_output=(
            "Brand safety tier, vendor status (active/inactive), "
            "last blocklist update timestamp, any configuration gaps."
        ),
        agent=agents["fraud_brand_safety"],
        context=[fraud_pull],
    )

    campaign_impact = Task(
        description=(
            f"Assess whether campaigns for {ctx.get('campaign_id', 'the advertiser')} "
            f"should be paused immediately given the IVT spike. "
            f"Check current campaign status and performance impact."
        ),
        expected_output=(
            "Campaign status, estimated financial impact (wasted spend), "
            "recommendation to pause or continue with blocklist update."
        ),
        agent=agents["campaign"],
        context=[fraud_pull, placement_audit],
    )

    escalation = Task(
        description=(
            "Based on all findings, make the escalation decision: "
            "PAUSE_NOW (IVT > 15% or >$500 wasted), "
            "MONITOR (IVT 5-15% with improving trend), or "
            "SAFE (IVT < 5%, isolated incident)."
        ),
        expected_output=(
            "Escalation decision: PAUSE_NOW | MONITOR | SAFE with justification. "
            "If PAUSE_NOW: list campaigns to pause immediately. "
            "If MONITOR: define monitoring interval and threshold. "
            "Required actions regardless of decision."
        ),
        agent=agents["triage"],
        context=[fraud_pull, placement_audit, brand_safety_check, campaign_impact],
    )

    return [fraud_pull, placement_audit, brand_safety_check, campaign_impact, escalation]


# ── Use Case 4: PMP Deal Zero Bids ───────────────────────────────────────────
def uc4_pmp_deal_tasks(agents: dict, ctx: dict) -> list[Task]:
    context_str = _ctx_str(ctx)

    deal_status = Task(
        description=(
            f"A PMP deal has zero bid requests. Check deal status and sync state for "
            f"{ctx.get('deal_id', 'the deal')}. "
            f"Pull bid stream data for the last 6 hours.\n\nContext:\n{context_str}"
        ),
        expected_output=(
            "Deal sync state (synced/pending/error), bid request count (last 6h), "
            "activation status, last sync timestamp."
        ),
        agent=agents["deal_inventory"],
    )

    campaign_targeting = Task(
        description=(
            f"Verify that active campaigns are correctly targeting deal "
            f"{ctx.get('deal_id', 'the deal')}. "
            f"Check if campaigns are active and have sufficient budget."
        ),
        expected_output=(
            "List of campaigns targeting this deal, their status, budget remaining, "
            "and bid settings. Confirm targeting is correctly configured."
        ),
        agent=agents["campaign"],
        context=[deal_status],
    )

    seat_mapping = Task(
        description=(
            f"Check buyer seat ID mapping for deal {ctx.get('deal_id', 'the deal')}. "
            f"Verify the seat ID configured on the deal matches the DSP registered seat ID."
        ),
        expected_output=(
            "Deal seat ID vs DSP registered seat ID. Match: YES/NO. "
            "If mismatch: exact values and correction steps."
        ),
        agent=agents["deal_inventory"],
        context=[deal_status],
    )

    kb_lookup = Task(
        description="Search the knowledge base for 'PMP zero bids' troubleshooting documentation.",
        expected_output="Relevant KB doc with step-by-step PMP zero-bid diagnosis checklist.",
        agent=agents["triage"],
        context=[deal_status],
    )

    fix_checklist = Task(
        description=(
            "Generate a step-by-step fix checklist for the PMP zero-bid issue "
            "based on all findings from deal status, campaign targeting, and seat mapping checks."
        ),
        expected_output=(
            "Numbered fix checklist (e.g. 1. Re-sync deal in DSP, 2. Update seat ID to X, "
            "3. Verify campaign targeting, 4. Wait 2h for bid stream to appear). "
            "Estimated time to resolution."
        ),
        agent=agents["deal_inventory"],
        context=[deal_status, campaign_targeting, seat_mapping, kb_lookup],
    )

    return [deal_status, campaign_targeting, seat_mapping, kb_lookup, fix_checklist]


# ── Use Case 5: Pixel & Conversion Attribution Drop ──────────────────────────
def uc5_pixel_attribution_tasks(agents: dict, ctx: dict) -> list[Task]:
    context_str = _ctx_str(ctx)

    pixel_check = Task(
        description=(
            f"Conversion tracking has dropped significantly. Check pixel fire log for "
            f"{ctx.get('pixel_id', 'the pixel')} (last 24h) and attribution window settings "
            f"for {ctx.get('campaign_id', 'the campaign')}.\n\nContext:\n{context_str}"
        ),
        expected_output=(
            "Pixel fire count (last 24h), attribution window (click-through + view-through), "
            "any anomalies in user agents or IP ranges."
        ),
        agent=agents["pixel_attribution"],
    )

    match_rate = Task(
        description=(
            f"Check cookie/device ID match rate for pixel {ctx.get('pixel_id', 'the pixel')}. "
            f"Determine if ITP or cookieless browsing is causing attribution failures."
        ),
        expected_output=(
            "Match rate percentage, breakdown by browser/device type. "
            "If match rate < 30%: ITP likely. If match rate 30-60%: partial ITP. "
            "Historical trend over last 30 days."
        ),
        agent=agents["pixel_attribution"],
        context=[pixel_check],
    )

    kb_lookup = Task(
        description="Search the knowledge base for 'ITP pixel attribution' and 'iOS 17 tracking' documentation.",
        expected_output="KB articles on ITP mitigation, server-side tagging, and first-party data solutions.",
        agent=agents["triage"],
        context=[pixel_check, match_rate],
    )

    campaign_config = Task(
        description=(
            f"Check if view-through conversions are enabled for "
            f"{ctx.get('campaign_id', 'the campaign')} and verify attribution windows "
            f"are correctly configured for a cookieless environment."
        ),
        expected_output=(
            "View-through conversion status (enabled/disabled), "
            "click-through window, view-through window, "
            "recommendation for cookieless-compatible settings."
        ),
        agent=agents["campaign"],
        context=[pixel_check, match_rate],
    )

    mitigation = Task(
        description=(
            "Generate a mitigation plan for the conversion attribution drop "
            "based on all findings. Prioritize solutions from immediate (hours) to strategic (weeks)."
        ),
        expected_output=(
            "Tiered mitigation plan:\n"
            "- Immediate: enable view-through conversions, adjust attribution windows\n"
            "- Short-term: implement server-side tagging\n"
            "- Strategic: first-party data integration, cookieless identity solution\n"
            "Estimated recovery of attribution with each step."
        ),
        agent=agents["pixel_attribution"],
        context=[pixel_check, match_rate, kb_lookup, campaign_config],
    )

    return [pixel_check, match_rate, kb_lookup, campaign_config, mitigation]


# ── Use Case 6: Pre-Launch Campaign Audit (Parallel Sweep) ───────────────────
def uc6_pre_launch_audit_tasks(agents: dict, ctx: dict) -> list[Task]:
    context_str = _ctx_str(ctx)

    # All 6 audit tasks run asynchronously (in parallel within CrewAI)
    campaign_audit = Task(
        description=(
            f"Pre-launch audit: Check campaign {ctx.get('campaign_id', 'the campaign')} — "
            f"pacing mode set? budget sufficient? bid above floor price?\n\nContext:\n{context_str}"
        ),
        expected_output="✅/❌ Campaign: status, budget, bid vs floor price.",
        agent=agents["campaign"],
        async_execution=True,
    )

    audience_audit = Task(
        description=(
            f"Pre-launch audit: Check segment and creative for {ctx.get('campaign_id', 'the campaign')} — "
            f"segment approved and size > 0? All creatives approved?"
        ),
        expected_output="✅/❌ Segment: size and status. ✅/❌ Creative: review status for each asset.",
        agent=agents["audience_creative"],
        async_execution=True,
    )

    pixel_audit = Task(
        description=(
            f"Pre-launch audit: Check pixel and attribution for {ctx.get('campaign_id', 'the campaign')} — "
            f"pixel firing correctly? Attribution window configured?"
        ),
        expected_output="✅/❌ Pixel: firing status. ✅/❌ Attribution: window settings. ⚠️ Match rate if < 50%.",
        agent=agents["pixel_attribution"],
        async_execution=True,
    )

    deal_audit = Task(
        description=(
            f"Pre-launch audit: If PMP campaign, check deal sync and seat ID for "
            f"{ctx.get('deal_id', 'any associated deal')}. If open auction, confirm and skip."
        ),
        expected_output="✅/❌ Deal: sync status, seat ID match. Or: N/A (open auction).",
        agent=agents["deal_inventory"],
        async_execution=True,
    )

    fraud_audit = Task(
        description=(
            f"Pre-launch audit: Check brand safety configuration — tier set? Blocklists active? "
            f"Third-party verification vendor enabled for {ctx.get('line_item_id', 'the line item')}?"
        ),
        expected_output="✅/❌ Brand safety: tier, IAS/DV status, blocklist last updated.",
        agent=agents["fraud_brand_safety"],
        async_execution=True,
    )

    kb_audit = Task(
        description="Search the knowledge base for 'campaign delivery checklist' and 'pre-launch requirements'.",
        expected_output="Pre-launch checklist from KB with any commonly missed items.",
        agent=agents["triage"],
        async_execution=True,
    )

    # Synthesis waits for all 6 parallel audits
    go_nogo = Task(
        description=(
            "Review all 6 audit results and issue a GO / NO-GO decision. "
            "Format the output as a checklist with ✅ / ❌ / ⚠️ per item, "
            "then a clear GO or NO-GO verdict with the specific issues that must be fixed."
        ),
        expected_output=(
            "Checklist report:\n"
            "✅/❌/⚠️ Campaign: ...\n"
            "✅/❌/⚠️ Creative: ...\n"
            "✅/❌/⚠️ Segment: ...\n"
            "✅/❌/⚠️ Pixel: ...\n"
            "✅/❌/⚠️ Brand Safety: ...\n"
            "✅/❌/⚠️ Deal (if PMP): ...\n\n"
            "→ GO / NO-GO: [reason + items to fix before launch]"
        ),
        agent=agents["triage"],
        context=[campaign_audit, audience_audit, pixel_audit, deal_audit, fraud_audit, kb_audit],
    )

    return [campaign_audit, audience_audit, pixel_audit, deal_audit, fraud_audit, kb_audit, go_nogo]


# ── Use Case 7: Budget Pacing Optimization ───────────────────────────────────
def uc7_budget_pacing_tasks(agents: dict, ctx: dict) -> list[Task]:
    context_str = _ctx_str(ctx)

    spend_curve = Task(
        description=(
            f"Campaign {ctx.get('campaign_id', 'the campaign')} is burning 80% of daily budget by 10am. "
            f"Pull hourly spend curve (last 3 days) and pacing settings. "
            f"Identify when budget exhaustion occurs each day.\n\nContext:\n{context_str}"
        ),
        expected_output=(
            "Hourly spend curve showing budget hit time each day, pacing mode (ASAP/EVEN/AHEAD), "
            "daily budget amount, frequency cap settings."
        ),
        agent=agents["campaign"],
    )

    audience_size = Task(
        description=(
            f"Check if audience size is too small for {ctx.get('campaign_id', 'the campaign')}, "
            f"which would cause rapid inventory exhaustion. Review segment status and size."
        ),
        expected_output=(
            "Segment size, reach estimate vs daily budget, "
            "assessment of whether audience is too narrow causing rapid depletion."
        ),
        agent=agents["audience_creative"],
        context=[spend_curve],
    )

    performance = Task(
        description=(
            f"Analyze if overpacing is hurting efficiency for {ctx.get('campaign_id', 'the campaign')}. "
            f"Check CTR, ROAS, and CPC — is early-day inventory premium quality or worse than average?"
        ),
        expected_output=(
            "CTR, ROAS, CPC for first 4 hours vs rest of day. "
            "Efficiency comparison: is overpacing causing premium CPM inventory to be wasted "
            "when cheaper inventory is available later in the day?"
        ),
        agent=agents["campaign"],
        context=[spend_curve],
    )

    recommendations = Task(
        description=(
            "Generate bid, frequency cap, and pacing mode recommendations to fix the overpacing issue. "
            "Provide specific numeric values for each change."
        ),
        expected_output=(
            "Specific recommendations with exact values:\n"
            "1. Pacing mode: change ASAP → EVEN\n"
            "2. Bid adjustment: reduce from $X to $Y to slow early burn\n"
            "3. Frequency cap: set to N impressions per user per day\n"
            "4. If audience too small: expand targeting by [specific criteria]\n"
            "Expected outcome: budget spread evenly through the day."
        ),
        agent=agents["campaign"],
        context=[spend_curve, audience_size, performance],
    )

    return [spend_curve, audience_size, performance, recommendations]
