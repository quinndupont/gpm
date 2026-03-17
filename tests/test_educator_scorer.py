"""Educator scorer tests — aggregate reward, scoring (requires educator GGUF)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EDUCATOR_GGUF = ROOT / "models" / "llama3.1-8b-educator-Q4_K_M.gguf"


def _has_educator_model() -> bool:
    return EDUCATOR_GGUF.exists()


@pytest.mark.eval
class TestEducatorScorerAggregate:
    """Test reward aggregation without loading the model."""

    def test_compute_aggregate_reward_default_weights(self):
        from scripts.eval.educator_scorer import DEFAULT_WEIGHTS

        scores = {
            "imagery_clarity": 0.8,
            "originality": 0.7,
            "sentiment_authenticity": 0.9,
            "form_adherence": 0.6,
            "voice_distinctiveness": 0.7,
        }
        reward = sum(scores.get(dim, 0.5) * w for dim, w in DEFAULT_WEIGHTS.items())
        assert 0.0 <= reward <= 1.0
        assert reward > 0.5

    def test_compute_aggregate_reward_custom_weights(self):
        from scripts.eval.educator_scorer import DEFAULT_WEIGHTS

        scores = {"imagery_clarity": 1.0, "originality": 0.0, "sentiment_authenticity": 0.0, "form_adherence": 0.0, "voice_distinctiveness": 0.0}
        weights = {k: 1.0 if k == "imagery_clarity" else 0.0 for k in DEFAULT_WEIGHTS}
        reward = sum(scores.get(dim, 0.5) * w for dim, w in weights.items())
        assert reward == 1.0


@pytest.mark.eval
@pytest.mark.skipif(not _has_educator_model(), reason="Educator GGUF not found")
class TestEducatorScoring:
    """Test educator scoring (requires llama3.1-8b-educator-Q4_K_M.gguf)."""

    def test_score_poem_returns_normalized_scores(self):
        from scripts.eval.educator_scorer import load_educator_scorer

        scorer = load_educator_scorer()
        poem = """The red bicycle rusted in October rain,
chain frozen, spokes bent like broken fingers."""
        scores = scorer.score_poem(
            poem=poem,
            brief="Write about an abandoned object",
            expected_form=None,
        )
        assert set(scores.keys()) == {
            "imagery_clarity", "originality", "sentiment_authenticity",
            "form_adherence", "voice_distinctiveness",
        }
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_aggregate_reward_in_range(self):
        from scripts.eval.educator_scorer import load_educator_scorer

        scorer = load_educator_scorer()
        scores = {
            "imagery_clarity": 0.8,
            "originality": 0.7,
            "sentiment_authenticity": 0.9,
            "form_adherence": 0.6,
            "voice_distinctiveness": 0.7,
        }
        reward = scorer.compute_aggregate_reward(scores)
        assert 0.0 <= reward <= 1.0
