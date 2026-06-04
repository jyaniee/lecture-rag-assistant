"""
품질 실험: §9 유형별 질문 실행, retrieval 튜닝( LLM 없이 ), JSON 저장.

터미널:
  python -m src.quality_experiments
  python -m src.quality_experiments --probe-only
  python -m src.quality_experiments --sweep-threshold
  python -m src.quality_experiments --sweep-top-k
  python -m src.quality_experiments --out reports/quality.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import RELEVANCE_SCORE_THRESHOLD, RETRIEVER_TOP_K
from src.rag_chain import ask_question, run_retrieval


QUALITY_TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "1-1",
        "category": "정상",
        "question": "RAG가 환각을 줄이는 원리를 설명해줘",
        "expect_rejected": False,
    },
    {
        "id": "1-2",
        "category": "정상",
        "question": "벡터 DB가 필요한 이유를 설명해줘",
        "expect_rejected": False,
    },
    {
        "id": "2-1",
        "category": "애매",
        "question": "RAG",
        "expect_rejected": None,
    },
    {
        "id": "3-1",
        "category": "없는",
        "question": "트랜스포머 attention을 수식으로 증명해줘",
        "expect_rejected": None,
    },
    {
        "id": "5-1",
        "category": "무관",
        "question": "오늘 서울 날씨 어때?",
        "expect_rejected": True,
    },
    {
        "id": "5-2",
        "category": "무관",
        "question": "맛있는 라면 레시피 알려줘",
        "expect_rejected": True,
    },
]

DEFAULT_THRESHOLD_CANDIDATES = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
DEFAULT_TOP_K_CANDIDATES = [2, 3, 4, 5, 6]


def _evaluate_case(result: Dict[str, Any], expect_rejected: Optional[bool]) -> str:
    if expect_rejected is None:
        return "manual"

    if result["is_rejected"] == expect_rejected:
        return "pass"

    return "fail"


def _row_from_result(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = result.get("retrieval") or {}
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "status": _evaluate_case(result, case.get("expect_rejected")),
        "is_rejected": result["is_rejected"],
        "rejection_reason": retrieval.get("rejection_reason"),
        "sources_count": len(result.get("sources", [])),
        "raw_hit_count": retrieval.get("raw_hit_count"),
        "context_chunk_count": retrieval.get("context_chunk_count"),
        "best_score": retrieval.get("best_score"),
        "threshold": retrieval.get("threshold"),
        "top_k": retrieval.get("top_k"),
        "search_k_used": retrieval.get("search_k_used"),
        "search_query": retrieval.get("search_query"),
    }


def run_quality_experiments(
    answer_mode: str = "기본 Q&A",
    cases: Optional[List[Dict[str, Any]]] = None,
    *,
    probe_only: bool = False,
    relevance_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """품질 실험을 실행하고 요약 dict를 반환합니다."""
    cases = cases or QUALITY_TEST_CASES
    results: List[Dict[str, Any]] = []

    for case in cases:
        if probe_only:
            result = run_retrieval(
                case["question"],
                top_k=top_k,
                relevance_threshold=relevance_threshold,
            )
            result["sources"] = []
        else:
            result = ask_question(
                case["question"],
                answer_mode,
                top_k=top_k,
                relevance_threshold=relevance_threshold,
            )

        results.append(_row_from_result(case, result))

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    manual = sum(1 for r in results if r["status"] == "manual")

    return {
        "mode": "probe" if probe_only else "full",
        "answer_mode": answer_mode,
        "relevance_threshold": relevance_threshold or RELEVANCE_SCORE_THRESHOLD,
        "top_k": top_k or RETRIEVER_TOP_K,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "manual_review": manual,
        "results": results,
    }


def run_threshold_sweep(
    candidates: Optional[List[float]] = None,
    cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """LLM 없이 threshold 후보별 자동 판정(pass/fail) 요약."""
    candidates = candidates or DEFAULT_THRESHOLD_CANDIDATES
    cases = cases or QUALITY_TEST_CASES
    auto_cases = [c for c in cases if c.get("expect_rejected") is not None]

    rows: List[Dict[str, Any]] = []
    for threshold in candidates:
        report = run_quality_experiments(
            cases=auto_cases,
            probe_only=True,
            relevance_threshold=threshold,
        )
        rows.append(
            {
                "threshold": threshold,
                "passed": report["passed"],
                "failed": report["failed"],
                "total": report["total"],
                "results": report["results"],
            }
        )

    best = max(rows, key=lambda r: (r["passed"], -r["failed"])) if rows else None

    return {
        "sweep": "threshold",
        "candidates": candidates,
        "rows": rows,
        "recommended": best["threshold"] if best else None,
        "note": "자동 판정 케이스만 집계. 애매·없는 유형은 full 실험 후 manual.",
    }


def run_top_k_sweep(
    candidates: Optional[List[int]] = None,
    cases: Optional[List[Dict[str, Any]]] = None,
    *,
    relevance_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """LLM 없이 top_k 후보별 context_chunk_count·거부율 요약."""
    candidates = candidates or DEFAULT_TOP_K_CANDIDATES
    cases = cases or QUALITY_TEST_CASES

    rows: List[Dict[str, Any]] = []
    for k in candidates:
        report = run_quality_experiments(
            cases=cases,
            probe_only=True,
            top_k=k,
            relevance_threshold=relevance_threshold,
        )
        rejected = sum(1 for r in report["results"] if r["is_rejected"])
        avg_chunks = _avg(
            r["context_chunk_count"]
            for r in report["results"]
            if r["context_chunk_count"] is not None
        )
        rows.append(
            {
                "top_k": k,
                "rejected_count": rejected,
                "avg_context_chunks": avg_chunks,
                "passed": report["passed"],
                "failed": report["failed"],
                "results": report["results"],
            }
        )

    return {
        "sweep": "top_k",
        "threshold": relevance_threshold or RELEVANCE_SCORE_THRESHOLD,
        "candidates": candidates,
        "rows": rows,
        "note": "k↑ 시 recall·토큰(생성) 모두 증가 경향. pass/fail은 자동 케이스만.",
    }


def _avg(values) -> Optional[float]:
    items = [v for v in values if v is not None]
    if not items:
        return None
    return round(sum(items) / len(items), 2)


def save_report(report: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **report,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_summary(report: Dict[str, Any]) -> None:
    if report.get("sweep") == "threshold":
        print("=== threshold sweep (probe only) ===")
        for row in report["rows"]:
            print(
                f"threshold={row['threshold']} pass={row['passed']} fail={row['failed']}"
            )
        print(f"recommended (auto cases): {report.get('recommended')}")
        return

    if report.get("sweep") == "top_k":
        print("=== top_k sweep (probe only) ===")
        for row in report["rows"]:
            print(
                f"k={row['top_k']} pass={row['passed']} fail={row['failed']} "
                f"rejected={row['rejected_count']} avg_chunks={row['avg_context_chunks']}"
            )
        return

    print(f"=== 품질 실험 ({report.get('mode', 'full')}) ===")
    print(
        f"threshold={report.get('relevance_threshold')} top_k={report.get('top_k')} "
        f"total={report['total']} pass={report['passed']} "
        f"fail={report['failed']} manual={report['manual_review']}"
    )
    for row in report["results"]:
        print(
            f"[{row['status']}] {row['id']} ({row['category']}) "
            f"rejected={row['is_rejected']} reason={row.get('rejection_reason')} "
            f"chunks={row['context_chunk_count']} best={row['best_score']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 품질 실험·튜닝")
    parser.add_argument("--probe-only", action="store_true", help="LLM 없이 검색만")
    parser.add_argument("--sweep-threshold", action="store_true", help="threshold 스윕")
    parser.add_argument("--sweep-top-k", action="store_true", help="top_k 스윕")
    parser.add_argument("--threshold", type=float, default=None, help="단일 실험 threshold")
    parser.add_argument("--top-k", type=int, default=None, help="단일 실험 top_k")
    parser.add_argument("--out", type=str, default=None, help="JSON 저장 경로")
    args = parser.parse_args()

    if args.sweep_threshold:
        report = run_threshold_sweep()
    elif args.sweep_top_k:
        report = run_top_k_sweep(relevance_threshold=args.threshold)
    else:
        report = run_quality_experiments(
            probe_only=args.probe_only,
            relevance_threshold=args.threshold,
            top_k=args.top_k,
        )

    _print_summary(report)

    if args.out:
        path = save_report(report, Path(args.out))
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
