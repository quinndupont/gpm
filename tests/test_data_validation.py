"""Data validation tests — schema, quality gate, chat format."""
import json
from pathlib import Path

import pytest

from scripts.data_generation.quality_gate import check as quality_gate_check

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


@pytest.mark.data
class TestSchemaValidation:
    """Schema tests for JSONL fixtures and data files."""

    def test_sample_critique_schema(self):
        data = _load_jsonl(FIXTURES_DIR / "sample_critique.jsonl")
        assert len(data) >= 1
        for entry in data:
            assert "poem" in entry
            assert "critique" in entry
            assert entry["poem"]
            assert entry["critique"]

    def test_sample_pairs_schema(self):
        data = _load_jsonl(FIXTURES_DIR / "sample_pairs.jsonl")
        assert len(data) >= 1
        for entry in data:
            assert "brief" in entry
            assert "poem" in entry
            assert entry["brief"]
            assert entry["poem"]

    def test_sample_rhyme_pairs_schema(self):
        data = _load_jsonl(FIXTURES_DIR / "sample_rhyme_pairs.jsonl")
        assert len(data) >= 1
        for entry in data:
            assert "brief" in entry
            assert "poem" in entry
            assert "form" in entry
            assert entry["brief"]
            assert entry["poem"]

    def test_sample_train_chat_format(self):
        data = _load_jsonl(FIXTURES_DIR / "sample_train.jsonl")
        assert len(data) >= 1
        for entry in data:
            assert "messages" in entry
            msgs = entry["messages"]
            assert len(msgs) >= 2
            roles = {m["role"] for m in msgs}
            assert "user" in roles
            assert "assistant" in roles
            for m in msgs:
                assert m["role"] in ("system", "user", "assistant")
                assert "content" in m
                assert m["content"]


@pytest.mark.data
class TestQualityGateIntegration:
    """Quality gate checks on educator-style data."""

    def test_quality_gate_passes_good_critique(self):
        entry = {
            "poem": "The cat sat on the mat.",
            "critique": (
                "The moment in line 1 holds attention. The rhyme works. "
                "This poem has found its shape.",
            ),
        }
        ok, reasons = quality_gate_check(entry)
        assert ok is True
        assert reasons == []

    def test_quality_gate_rejects_llm_ism(self):
        entry = {
            "poem": "Roses are red",
            "critique": (
                "This poem delves into the rich tapestry of emotion. "
                "It's worth noting how it resonates deeply.",
            ),
        }
        ok, reasons = quality_gate_check(entry)
        assert ok is False
        assert "llm_ism" in reasons

    def test_quality_gate_rejects_voice_consistency(self):
        entry = {
            "poem": "Roses are red",
            "critique": "Nice use of imagery! Great job!",
        }
        ok, reasons = quality_gate_check(entry)
        assert ok is False
        assert "voice_consistency" in reasons

    def test_quality_gate_rejects_empty(self):
        entry = {"poem": "x", "critique": ""}
        ok, reasons = quality_gate_check(entry)
        assert ok is False
        assert "empty response" in reasons

    def test_quality_gate_on_fixture_critiques(self):
        data = _load_jsonl(FIXTURES_DIR / "sample_critique.jsonl")
        passed = 0
        for entry in data:
            ok, _ = quality_gate_check(entry)
            if ok:
                passed += 1
        assert passed >= 1, "At least one fixture critique should pass the quality gate"


@pytest.mark.data
class TestNoEmptyFields:
    """No blank poem, critique, brief, etc."""

    def test_fixtures_no_empty_content(self):
        fnames = [
            "sample_critique.jsonl", "sample_pairs.jsonl",
            "sample_rhyme_pairs.jsonl", "sample_train.jsonl",
        ]
        for fname in fnames:
            data = _load_jsonl(FIXTURES_DIR / fname)
            for entry in data:
                if "messages" in entry:
                    for m in entry["messages"]:
                        assert m.get("content", "").strip(), f"Empty content in {fname}"
                else:
                    for key in ["poem", "critique", "brief"]:
                        if key in entry:
                            val = entry[key]
                        assert (
                            val.strip() if isinstance(val, str) else val
                        ), f"Empty {key} in {fname}"


@pytest.mark.data
class TestRhymeDensityGate:
    """Rhyme training data density gate (strict_rhyme_density >= 0.6)."""

    @pytest.fixture
    def rhyme_train_path(self):
        return DATA_DIR / "rhyme_training" / "train.jsonl"

    def test_rhyme_train_density_if_exists(self, rhyme_train_path):
        if not rhyme_train_path.exists():
            pytest.skip("data/rhyme_training/train.jsonl not present")
        from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme
        data = _load_jsonl(rhyme_train_path)
        for entry in data:
            msgs = entry.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if not assistant.strip():
                continue
            if "rhyme" not in user.lower() and "scheme" not in user.lower():
                continue
            result = analyze_rhyme(assistant)
            density = result.get("strict_rhyme_density", 0)
            assert density >= 0.6, f"Rhyme poem has strict_rhyme_density={density}, need >= 0.6"
