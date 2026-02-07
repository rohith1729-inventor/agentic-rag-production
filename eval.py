#!/usr/bin/env python3
"""
Evaluation script for the RAG system.

Sends test questions to the /query endpoint, collects responses, and
produces a summary report saved to reports/eval_report.md.

Usage:
    python eval.py              # default: http://localhost:8000
    python eval.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

TEST_QUESTIONS: List[str] = [
    # Answerable from data/sample.md -----------------------------------------
    "What is a RAG system?",
    "What are the components of the ingestion pipeline?",
    "What embedding model does the system use?",
    "What vector database is used in this project?",
    "How does the query pipeline work?",
    "What is FlashRank used for?",
    "What are the three answer types the system supports?",
    "What LLM provider does the system use for synthesis?",
    "What is the chunk size used during ingestion?",
    "What metadata is stored with each chunk in Qdrant?",
    "What framework is the API built with?",
    "What Python version is required?",
    # NOT_FOUND questions (unrelated topics) ----------------------------------
    "How do you make a chocolate souffle?",
    "What are the rules of cricket?",
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of Mongolia?",
    "How does photosynthesis work?",
    "What is the airspeed velocity of an unladen swallow?",
    "Explain the theory of general relativity.",
    "What are the best hiking trails in Patagonia?",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(base_url: str) -> Dict[str, Any]:
    """Send all test questions and collect results."""
    query_url = f"{base_url.rstrip('/')}/query"
    results: List[Dict[str, Any]] = []

    with httpx.Client(timeout=60.0) as client:
        for idx, question in enumerate(TEST_QUESTIONS, 1):
            print(f"[{idx}/{len(TEST_QUESTIONS)}] {question}")
            try:
                resp = client.post(query_url, json={"question": question})
                resp.raise_for_status()
                data = resp.json()
                results.append({
                    "question": question,
                    "answer_type": data.get("answer_type", "UNKNOWN"),
                    "citations": data.get("citations", []),
                    "timings_ms": data.get("timings_ms", {}),
                    "answer": data.get("answer", ""),
                    "error": None,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")
                results.append({
                    "question": question,
                    "answer_type": "ERROR",
                    "citations": [],
                    "timings_ms": {},
                    "answer": "",
                    "error": str(exc),
                })

    return _build_report(results)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics and return a report dict."""
    total = len(results)
    complete = sum(1 for r in results if r["answer_type"] == "COMPLETE")
    partial = sum(1 for r in results if r["answer_type"] == "PARTIAL")
    not_found = sum(1 for r in results if r["answer_type"] == "NOT_FOUND")
    errors = sum(1 for r in results if r["answer_type"] == "ERROR")

    # Citation presence: among non-NOT_FOUND, non-ERROR responses
    answerable = [
        r for r in results
        if r["answer_type"] in ("COMPLETE", "PARTIAL")
    ]
    with_citations = sum(1 for r in answerable if len(r["citations"]) > 0)

    # Latency
    latencies = [
        r["timings_ms"].get("total", 0)
        for r in results
        if r["timings_ms"].get("total") is not None and r["answer_type"] != "ERROR"
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "complete": complete,
        "complete_rate": round(complete / total * 100, 1) if total else 0,
        "partial": partial,
        "partial_rate": round(partial / total * 100, 1) if total else 0,
        "not_found": not_found,
        "not_found_rate": round(not_found / total * 100, 1) if total else 0,
        "errors": errors,
        "answerable_count": len(answerable),
        "with_citations": with_citations,
        "avg_latency_ms": round(avg_latency, 1),
        "results": results,
    }

    return report


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def write_report_md(report: Dict[str, Any], path: str) -> None:
    """Write the report dict as a Markdown file."""
    lines: List[str] = []
    lines.append("# RAG Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {report['timestamp']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total questions | {report['total']} |")
    lines.append(f"| COMPLETE | {report['complete']} ({report['complete_rate']}%) |")
    lines.append(f"| PARTIAL | {report['partial']} ({report['partial_rate']}%) |")
    lines.append(f"| NOT_FOUND | {report['not_found']} ({report['not_found_rate']}%) |")
    lines.append(f"| Errors | {report['errors']} |")
    lines.append(f"| Answerable with citations | {report['with_citations']} / {report['answerable_count']} |")
    lines.append(f"| Average total latency | {report['avg_latency_ms']} ms |")
    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| # | Question | Answer Type | Citations | Total Latency (ms) |")
    lines.append("|---|---|---|---|---|")

    for idx, r in enumerate(report["results"], 1):
        q = r["question"]
        at = r["answer_type"]
        cit = len(r["citations"])
        lat = r["timings_ms"].get("total", "N/A")
        lines.append(f"| {idx} | {q} | {at} | {cit} | {lat} |")

    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation runner")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the RAG API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    print(f"Running evaluation against {args.url}")
    print(f"Questions: {len(TEST_QUESTIONS)}\n")

    report = run_eval(args.url)

    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "reports",
        "eval_report.md",
    )
    write_report_md(report, report_path)

    # Print summary to stdout
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total:      {report['total']}")
    print(f"COMPLETE:   {report['complete']} ({report['complete_rate']}%)")
    print(f"PARTIAL:    {report['partial']} ({report['partial_rate']}%)")
    print(f"NOT_FOUND:  {report['not_found']} ({report['not_found_rate']}%)")
    print(f"Errors:     {report['errors']}")
    print(f"Citations:  {report['with_citations']}/{report['answerable_count']} answerable have citations")
    print(f"Avg latency: {report['avg_latency_ms']} ms")


if __name__ == "__main__":
    main()
