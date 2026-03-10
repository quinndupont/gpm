"""Prompt loader and template tests."""
import hashlib
import json
from pathlib import Path

import pytest

from models.prompts.loader import get_persona, get_prompt, render_prompt

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "models" / "prompts"
MANIFEST_PATH = Path(__file__).resolve().parent / "fixtures" / "prompt_manifest.json"


@pytest.mark.prompts
class TestPersonaLoading:
    """Persona loading and structure."""

    def test_get_persona_educator_neutral(self):
        text = get_persona("educator_neutral")
        assert isinstance(text, str)
        assert len(text) > 50
        assert "poetry educator" in text.lower()
        assert "This poem has found its shape" in text

    def test_get_persona_educator_condensed(self):
        text = get_persona("educator_condensed")
        assert isinstance(text, str)
        assert len(text) > 20

    def test_get_persona_poet(self):
        text = get_persona("poet")
        assert isinstance(text, str)
        assert "poet" in text.lower()

    def test_get_persona_poet_rhyme(self):
        text = get_persona("poet_rhyme")
        assert isinstance(text, str)
        assert "rhyme" in text.lower()

    def test_get_persona_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            get_persona("nonexistent")


@pytest.mark.prompts
class TestPromptLoading:
    """Prompt loading and template round-trip."""

    @pytest.mark.parametrize("category,prompt_id,template", [
        ("tuning", "critique", "default"),
        ("tuning", "brief", "default"),
        ("tuning", "poet_generation", "default"),
        ("tuning", "poet_generation", "rhyme"),
        ("tuning", "poet_generation", "user_suffix"),
        ("tuning", "poet_generation", "rhyme_suffix"),
        ("tuning", "autopsy", "default"),
        ("tuning", "comparison", "default"),
        ("tuning", "lesson", "default"),
        ("tuning", "revision_brief", "default"),
        ("tuning", "rhyme_pairs", "brief"),
        ("tuning", "rhyme_pairs", "poet"),
        ("tuning", "rhyme_pairs", "critique"),
        ("tuning", "approval", "approve"),
        ("tuning", "approval", "reject"),
        ("tuning", "dialogue", "student_revision"),
        ("tuning", "dialogue", "dialogue"),
        ("inference", "brief", "default"),
        ("inference", "critique", "default"),
        ("inference", "poet_generation", "default"),
    ])
    def test_get_prompt_succeeds(self, category, prompt_id, template):
        tpl = get_prompt(category, prompt_id, template)
        assert isinstance(tpl, str)
        assert len(tpl) > 50

    def test_get_prompt_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            get_prompt("tuning", "nonexistent", "default")

    def test_get_prompt_bad_template_raises(self):
        with pytest.raises(KeyError):
            get_prompt("tuning", "critique", "nonexistent")


@pytest.mark.prompts
class TestTemplateRendering:
    """Template rendering with sample kwargs."""

    def test_render_critique(self):
        out = render_prompt("tuning", "critique", "default", poem_text="The cat sat on the mat.")
        assert "The cat sat on the mat." in out
        assert "workshop" in out.lower()

    def test_render_brief(self):
        out = render_prompt("tuning", "brief", "default", user_request="Write a poem about winter")
        assert "Write a poem about winter" in out

    def test_render_poet_generation(self):
        out = render_prompt(
            "tuning", "poet_generation", "default",
            brief="Write a quatrain about rain.",
        )
        assert "Write a quatrain about rain." in out
        assert "Output ONLY the poem" in out

    def test_render_autopsy(self):
        out = render_prompt("tuning", "autopsy", "default", bad_poem_text="Roses are red")
        assert "Roses are red" in out

    def test_render_comparison(self):
        out = render_prompt(
            "tuning", "comparison", "default",
            poem_a="Poem A text", poem_b="Poem B text",
        )
        assert "Poem A text" in out
        assert "Poem B text" in out

    def test_render_lesson(self):
        out = render_prompt("tuning", "lesson", "default", question="What is enjambment?")
        assert "What is enjambment?" in out

    def test_render_revision_brief(self):
        out = render_prompt("tuning", "revision_brief", "default",
            poem_text="Original poem", critique="Your critique")
        assert "Original poem" in out
        assert "Your critique" in out

    def test_render_rhyme_pairs_brief(self):
        out = render_prompt("tuning", "rhyme_pairs", "brief",
            user_request="Write a poem", form_desc="limerick (AABBA)")
        assert "Write a poem" in out
        assert "limerick" in out

    def test_render_rhyme_pairs_critique(self):
        out = render_prompt("tuning", "rhyme_pairs", "critique",
            poem="Poem text", form_name="limerick", expected_scheme="AABBA",
            analysis_block="Analysis here")
        assert "Poem text" in out
        assert "AABBA" in out

    def test_render_approval_approve(self):
        out = render_prompt("tuning", "approval", "approve", poem="Poem", analysis="Analysis")
        assert "Poem" in out
        assert "This poem has found its shape" in out

    def test_render_approval_reject(self):
        out = render_prompt("tuning", "approval", "reject",
            poem="Poem", form_desc="sonnet", expected_scheme="ABAB", analysis="Analysis")
        assert "Poem" in out
        assert "Do NOT end" in out

    def test_render_inference_brief(self):
        out = render_prompt("inference", "brief", "default",
            user_request="Write a poem", style_ctx="")
        assert "Write a poem" in out

    def test_render_inference_critique(self):
        out = render_prompt("inference", "critique", "default",
            brief="Brief", draft="Draft", history_ctx="", form_ctx="")
        assert "Brief" in out
        assert "Draft" in out

    def test_render_inference_poet_generation(self):
        out = render_prompt("inference", "poet_generation", "default",
            brief="Brief", scheme_reminder="")
        assert "Brief" in out

    def test_render_dialogue_student_revision(self):
        out = render_prompt("tuning", "dialogue", "student_revision",
            poem_text="Original", critique="Your critique")
        assert "Original" in out
        assert "Your critique" in out

    def test_render_dialogue_dialogue(self):
        out = render_prompt("tuning", "dialogue", "dialogue",
            poem_text="Original", critique="Critique", revised_poem="Revised")
        assert "Original" in out
        assert "Revised" in out


@pytest.mark.prompts
class TestRegressionSnapshots:
    """Regression snapshots for critical prompts — catch accidental drift."""

    def test_critique_snapshot(self):
        out = render_prompt("tuning", "critique", "default", poem_text="The cat sat on the mat.")
        assert "The cat sat on the mat." in out
        assert "workshop" in out.lower()
        assert "This poem has found its shape" in out
        assert "What's alive" in out or "what's alive" in out

    def test_brief_snapshot(self):
        out = render_prompt("tuning", "brief", "default", user_request="Write a poem about grief.")
        assert "Write a poem about grief." in out
        assert "Angle" in out
        assert "Clichés" in out or "clichés" in out

    def test_poet_generation_snapshot(self):
        out = render_prompt(
            "tuning", "poet_generation", "default",
            brief="Write a quatrain.",
        )
        assert "Write a quatrain." in out
        assert "Output ONLY the poem" in out


@pytest.mark.prompts
class TestPromptManifest:
    """Snapshot manifest — prompt drift detection."""

    def test_prompt_manifest_matches(self):
        manifest = json.loads(MANIFEST_PATH.read_text())
        for rel_path, expected_hash in manifest.items():
            full_path = PROMPTS_DIR / rel_path
            assert full_path.exists(), f"Prompt file {rel_path} missing"
            current_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()[:16]
            assert current_hash == expected_hash, (
                f"Prompt {rel_path} changed. Update tests/fixtures/prompt_manifest.json "
                f"with hash {current_hash}",
            )

    def test_no_orphan_prompts(self):
        manifest = json.loads(MANIFEST_PATH.read_text())
        for p in sorted(PROMPTS_DIR.rglob("*.json")):
            rel = str(p.relative_to(PROMPTS_DIR))
            assert rel in manifest, (
                f"New prompt {rel} not in manifest. Add it with: "
                "hashlib.sha256(p.read_bytes()).hexdigest()[:16]",
            )
