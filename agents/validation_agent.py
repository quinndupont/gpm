"""
ValidationAgent: Filters and validates (prompt, poem) pairs for generator training.
Removes low-quality, duplicate, or malformed poems.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

from nlp_dedup import Deduper

from .base_agent import GPMAgent


class ValidationAgent(GPMAgent):
    """
    Filters and validates (prompt, poem) pairs for poetry generator training.
    """

    def __init__(self, config: Dict):
        super().__init__("validation_agent", config)
        val_config = config.get("validation", config)
        self.min_poem_length = val_config.get("min_poem_length", 50)
        self.min_lines = val_config.get("min_lines", 2)
        self.check_repetition = val_config.get("check_repetition", True)

    # Preamble patterns from Ollama/LLM reverse-prompt generation
    _PREAMBLE_PATTERNS = re.compile(
        r'^(here are|sure!|i\'d be happy|certainly|of course|great!|let me|absolutely|'
        r'here\'s|okay|alright|the following|below are|i\'ve generated)',
        re.IGNORECASE,
    )
    _CATEGORY_LABEL = re.compile(
        r'^\s*(?:prompt\s*\d+\s*)?\[(?:thematic|formal|stylistic|constraint|emotional|general)[^\]]*\]\s*:?\s*',
        re.IGNORECASE,
    )

    def _clean_prompt(self, prompt: str) -> str:
        """Strip LLM preamble and category labels from reverse-engineered prompts."""
        lines = prompt.strip().split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip preamble lines
            if self._PREAMBLE_PATTERNS.match(stripped):
                continue
            # Strip category labels like "[Thematic]:"
            stripped = self._CATEGORY_LABEL.sub('', stripped).strip()
            # Strip leading/trailing quotes
            if stripped.startswith('"') and stripped.endswith('"'):
                stripped = stripped[1:-1].strip()
            if stripped:
                cleaned.append(stripped)
        return ' '.join(cleaned) if cleaned else prompt.strip()

    def _has_repetition(self, text: str, threshold: int = 5) -> bool:
        """Detect if text has excessive repetition"""
        lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
        seen = {}
        for line in lines:
            key = line[:80] if len(line) > 80 else line
            if key:
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > threshold:
                    return True
        return False

    def validate_pair(self, pair: Dict) -> Tuple[bool, str]:
        """Check if (prompt, poem) pair meets quality standards"""
        if pair.get("error"):
            return False, f"Generation error: {pair['error']}"

        prompt = pair.get("prompt", "").strip()
        poem = pair.get("poem", "").strip()

        if not prompt:
            return False, "Missing prompt"
        if not poem:
            return False, "Missing poem"

        # Clean prompt (strip LLM preamble/labels)
        cleaned_prompt = self._clean_prompt(prompt)
        if not cleaned_prompt:
            return False, "Prompt empty after cleaning"
        pair["prompt"] = cleaned_prompt

        # Reject if prompt is longer than the poem (likely model commentary)
        if len(cleaned_prompt) > len(poem):
            return False, "Prompt longer than poem (likely model commentary)"

        if len(poem) < self.min_poem_length:
            return False, f"Poem too short: {len(poem)} chars"

        lines = [l for l in poem.split("\n") if l.strip()]
        if len(lines) < self.min_lines:
            return False, f"Poem has too few lines: {len(lines)}"

        if self.check_repetition and self._has_repetition(poem):
            return False, "Excessive repetition detected"

        return True, "passed"

    def deduplicate_poems(self, pairs: List[Dict]) -> List[Dict]:
        """Remove near-duplicate poems using nlp_dedup (MinHash)."""
        if not pairs:
            return pairs

        dedup_config = self.config.get("validation", {}).get("dedup", {})
        deduper = Deduper(
            split_method=dedup_config.get("split_method", "word_ngram"),
            ngram_size=dedup_config.get("ngram_size", 13),
            similarity_threshold=dedup_config.get("similarity_threshold", 0.8),
            store_corpus_to_disk=False,
            store_mask_to_disk=False,
            store_lsh_cache_to_disk=False,
            store_config_to_disk=False,
            return_generator=True,
            verbose=False,
        )
        mask_gen = deduper.deduplicate(
            corpus=pairs,
            text_column="poem",
            output_dir="checkpoints/nlp_dedup_validation_tmp",
            overwrite=True,
            num_docs=len(pairs),
        )
        mask = list(mask_gen)
        return [pairs[i] for i, m in enumerate(mask) if not m["duplicate"]]

    def execute(self, input_data: Dict) -> Dict:
        """Validate and filter (prompt, poem) pairs"""
        input_file = input_data.get(
            "training_file", input_data.get("annotations_file", "data/gpm_generator_train.jsonl")
        )
        output_file = Path("data/training/gpm_validated.jsonl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pairs = []
        with open(input_file) as f:
            for line in f:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        self.logger.info(f"Loaded {len(pairs)} pairs for validation")

        valid = []
        invalid = []
        reasons = {}

        for pair in pairs:
            is_valid, reason = self.validate_pair(pair)
            if is_valid:
                valid.append(pair)
            else:
                invalid.append({**pair, "failure_reason": reason})
                reasons[reason] = reasons.get(reason, 0) + 1

        self.logger.info(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
        self.logger.info(f"Failure reasons: {reasons}")

        valid = self.deduplicate_poems(valid)
        self.logger.info(f"After deduplication: {len(valid)}")

        with open(output_file, "w") as f:
            for pair in valid:
                f.write(json.dumps(pair) + "\n")

        rejected_file = output_file.parent / "gpm_rejected.jsonl"
        with open(rejected_file, "w") as f:
            for pair in invalid:
                f.write(json.dumps(pair) + "\n")

        self.save_state({
            "status": "completed",
            "total_input": len(pairs),
            "valid": len(valid),
            "invalid": len(invalid),
            "rejection_reasons": reasons,
            "output_file": str(output_file),
        })

        return {
            "validated_file": str(output_file),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "acceptance_rate": len(valid) / len(pairs) if pairs else 0,
        }
