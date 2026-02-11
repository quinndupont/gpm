import json
import re
from pathlib import Path
from typing import List, Dict

from nlp_dedup import Deduper

from .base_agent import GPMAgent


class ValidationAgent(GPMAgent):
    """
    Filters and validates synthetic annotations
    Removes low-quality, duplicate, or malformed analyses
    """

    def __init__(self, config: Dict):
        super().__init__('validation_agent', config)
        val_config = config.get('validation', config)
        self.min_analysis_length = val_config.get('min_analysis_length', 200)
        self.min_rating = val_config.get('min_rating', 2)

    def _has_repetition(self, text: str, threshold: int = 5) -> bool:
        """Detect if text has excessive repetition"""
        sentences = re.split(r'[.!?]+', text)
        seen = {}
        for sent in sentences:
            normalized = sent.strip().lower()[:50]
            if normalized:
                seen[normalized] = seen.get(normalized, 0) + 1
                if seen[normalized] > threshold:
                    return True
        return False

    def validate_annotation(self, ann: Dict) -> tuple:
        """Check if annotation meets quality standards"""
        if ann.get('error'):
            return False, f"Generation error: {ann['error']}"

        # Support both processed format (analysis) and raw format (raw_response)
        analysis = ann.get('analysis') or ann.get('raw_response', '')

        # Check if this is a ranking task (has parsed_scores as array) vs analysis task
        parsed_scores = ann.get('parsed_scores')
        is_ranking_task = isinstance(parsed_scores, list) and len(parsed_scores) > 0

        if is_ranking_task:
            # Validation for ranking/comparative tasks
            if not analysis:
                return False, "Missing response"

            # For ranking tasks, just check that we have valid scores
            if not parsed_scores:
                return False, "No scores found in ranking task"

            # Validate that scores are in valid range
            for item in parsed_scores:
                if isinstance(item, list) and len(item) >= 3:
                    score = item[2]
                    if not isinstance(score, int) or score < 1 or score > 5:
                        return False, f"Invalid score in ranking: {score}"

            return True, "passed"

        # Validation for detailed analysis tasks
        if len(analysis) < self.min_analysis_length:
            return False, f"Analysis too short: {len(analysis)} chars"

        word_count = len(analysis.split())
        if word_count < 50:
            return False, f"Insufficient analysis: {word_count} words"

        if self._has_repetition(analysis):
            return False, "Excessive repetition detected"

        # Support both processed format (ratings) and raw format (parsed_scores)
        ratings = ann.get('ratings') or ann.get('parsed_scores', {})
        if isinstance(ratings, dict):
            for key, val in ratings.items():
                if isinstance(val, int) and (val < 1 or val > 5):
                    return False, f"Invalid rating for {key}: {val}"

        analysis_lower = analysis.lower()
        has_content = any(
            word in analysis_lower
            for word in ['poem', 'verse', 'stanza', 'metaphor', 'image', 'theme', 'meter', 'rhyme']
        )
        if not has_content:
            return False, "Analysis lacks poetic terminology"

        return True, "passed"

    def deduplicate_analyses(self, annotations: List[Dict]) -> List[Dict]:
        """Remove near-duplicate analyses using nlp_dedup (MinHash)."""
        if not annotations:
            return annotations

        # Normalize field names before deduplication
        # Support both 'analysis' and 'raw_response' fields
        normalized = []
        for ann in annotations:
            norm_ann = ann.copy()
            if 'raw_response' in ann and 'analysis' not in ann:
                norm_ann['analysis'] = ann['raw_response']
            normalized.append(norm_ann)

        dedup_config = self.config.get('validation', {}).get('dedup', {})
        deduper = Deduper(
            split_method=dedup_config.get('split_method', 'word_ngram'),
            ngram_size=dedup_config.get('ngram_size', 13),
            similarity_threshold=dedup_config.get('similarity_threshold', 0.8),
            store_corpus_to_disk=False,
            store_mask_to_disk=False,
            store_lsh_cache_to_disk=False,
            store_config_to_disk=False,
            return_generator=True,
            verbose=False,
        )
        mask_gen = deduper.deduplicate(
            corpus=normalized,
            text_column='analysis',
            output_dir='checkpoints/nlp_dedup_validation_tmp',
            overwrite=True,
            num_docs=len(normalized),
        )
        mask = list(mask_gen)
        # Return original annotations (not normalized) for the non-duplicates
        return [annotations[i] for i, m in enumerate(mask) if not m['duplicate']]

    def execute(self, input_data: Dict) -> Dict:
        """Validate and filter annotations"""
        input_file = input_data.get('annotations_file', 'data/annotated/gpm_annotations.jsonl')
        output_file = Path('data/training/gpm_validated.jsonl')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        annotations = []
        with open(input_file) as f:
            for line in f:
                try:
                    annotations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        self.logger.info(f"Loaded {len(annotations)} annotations for validation")

        valid = []
        invalid = []
        reasons = {}

        for ann in annotations:
            is_valid, reason = self.validate_annotation(ann)
            if is_valid:
                valid.append(ann)
            else:
                invalid.append({**ann, 'failure_reason': reason})
                reasons[reason] = reasons.get(reason, 0) + 1

        self.logger.info(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
        self.logger.info(f"Failure reasons: {reasons}")

        valid = self.deduplicate_analyses(valid)
        self.logger.info(f"After deduplication: {len(valid)}")

        with open(output_file, 'w') as f:
            for ann in valid:
                f.write(json.dumps(ann) + '\n')

        rejected_file = output_file.parent / 'gpm_rejected.jsonl'
        with open(rejected_file, 'w') as f:
            for ann in invalid:
                f.write(json.dumps(ann) + '\n')

        self.save_state({
            'status': 'completed',
            'total_input': len(annotations),
            'valid': len(valid),
            'invalid': len(invalid),
            'rejection_reasons': reasons,
            'output_file': str(output_file)
        })

        return {
            'validated_file': str(output_file),
            'valid_count': len(valid),
            'invalid_count': len(invalid),
            'acceptance_rate': len(valid) / len(annotations) if annotations else 0
        }
