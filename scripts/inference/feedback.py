#!/usr/bin/env python3
"""Feedback collection for retraining. S7"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_FILE = ROOT / "feedback" / "feedback.jsonl"


def record(pipeline_result: dict, user_feedback: dict):
    """Append feedback entry to feedback.jsonl."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_request": pipeline_result.get("user_request", ""),
        "final_poem": pipeline_result.get("final_poem", ""),
        "educator_note": pipeline_result.get("educator_note", ""),
        "generation_brief": pipeline_result.get("generation_brief", ""),
        "revision_count": len(pipeline_result.get("revision_history", [])),
        "revision_history": pipeline_result.get("revision_history", []),
        "models": pipeline_result.get("metadata", {}),
        "user_feedback": user_feedback,
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # Example usage
    record(
        {"user_request": "test", "final_poem": "", "educator_note": "", "generation_brief": "", "revision_history": [], "metadata": {}},
        {"rating": 3, "kept_poem": True, "notes": "", "educator_helpful": True, "changes": ""},
    )
    print(f"Recorded to {FEEDBACK_FILE}")
