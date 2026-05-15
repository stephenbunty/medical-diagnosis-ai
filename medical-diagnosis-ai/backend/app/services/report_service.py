"""Structured clinical report generation with optional LLM explanations."""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def build_structured_report(
    modality: str,
    labels: dict[str, float],
    severity_score: float,
    top_findings: list[str],
) -> dict[str, Any]:
    """
    Rule-based structured report (always available offline).
    LLM can extend `llm_explanation` when API key is configured.
    """
    findings_lines = [f"- {name}: probability {val:.2f}" for name, val in sorted(labels.items(), key=lambda x: -x[1])]
    findings = "Radiologic findings:\n" + "\n".join(findings_lines[:8])
    impression_parts = []
    for name, val in sorted(labels.items(), key=lambda x: -x[1])[:3]:
        if val >= 0.5:
            impression_parts.append(f"High suspicion for {name.replace('_', ' ')} ({val:.0%}).")
        elif val >= 0.35:
            impression_parts.append(f"Possible {name.replace('_', ' ')} ({val:.0%}).")
    if not impression_parts:
        impression_parts.append("No high-confidence abnormality detected by the model.")
    impression = " ".join(impression_parts)

    if severity_score >= 0.7:
        suggested = "Urgent clinical correlation and specialist referral recommended."
    elif severity_score >= 0.45:
        suggested = "Clinical correlation and follow-up imaging may be appropriate."
    else:
        suggested = "Routine follow-up per clinical judgment."

    return {
        "findings": findings,
        "impression": impression,
        "severity_score": float(severity_score),
        "suggested_action": suggested,
        "modality": modality,
        "summary_bullets": "\n".join(f"- {t}" for t in top_findings),
    }


async def maybe_llm_explanation(report_payload: dict[str, Any]) -> str | None:
    """Optional OpenAI call for patient-friendly / teaching explanations."""
    if not settings.openai_api_key:
        return None
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        prompt = (
            "You are a radiology teaching assistant. Given structured model output, "
            "write 3 short bullet points explaining what the doctor should verify next. "
            "Do not diagnose definitively. Data:\n"
            f"{report_payload}"
        )
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:  # pragma: no cover
        logger.warning("LLM explanation skipped: %s", e)
        return None
