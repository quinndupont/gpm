import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import ollama
import yaml

from .base_agent import GPMAgent


def load_cat_criteria() -> Dict:
    path = Path(__file__).parent.parent / "config" / "prompts" / "cat_criteria.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class AnnotationAgent(GPMAgent):
    """
    Generates synthetic poetry analysis using Ollama.
    Implements CAT methodology with configurable criteria (creativity, quality, innovativeness, similarity, poeticness).
    """

    def __init__(self, config: Dict):
        super().__init__('annotation_agent', config)
        self.cat_config = load_cat_criteria()
        ollama_config = config.get('ollama', config)
        self.model = ollama_config.get('model', 'llama3.2:3b')
        self.batch_size = ollama_config.get('batch_size') or self.cat_config.get('batch_size', 15)
        self.temperature = ollama_config.get('temperature') or self.cat_config.get('temperature', 1.0)
        self.max_tokens = ollama_config.get('max_tokens', 2000)
        self.criteria = list(self.cat_config.get('criteria', {}).keys())

        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(f"Ollama not available: {e}. Run 'ollama serve' first.")

    def create_cat_prompt(self, poems: List[Dict], criterion: str) -> str:
        """Create CAT prompt for given criterion and batch of poems."""
        if criterion not in self.cat_config.get('criteria', {}):
            raise ValueError(f"Unknown criterion: {criterion}")
        instruction = self.cat_config['criteria'][criterion]['instruction']
        output_format = self.cat_config.get('output_format', '')

        prompt = f"Below is the collection of {len(poems)} poems. {instruction}\n\n{output_format}\n\nPOEMS:\n\n"
        for i, poem in enumerate(poems, 1):
            prompt += f"--- POEM {i}: \"{poem['title']}\" by {poem['author']} ---\n"
            text = poem['text']
            if len(text) > 1500:
                text = text[:1500] + "\n[...truncated...]"
            prompt += text + "\n\n"
        return prompt

    def _parse_ranked_list(self, raw: str) -> Dict[tuple, int]:
        """
        Parse ranked list format: "N. Author - Title : score"
        Returns dict mapping (author, title) -> score.

        Author/title may have slight variations; we match by normalizing.
        """
        scores = {}
        # Match: position. Author - Title : score
        pattern = r'\d+\.\s*(.+?)\s*-\s*(.+?)\s*:\s*(\d)'
        for m in re.finditer(pattern, raw):
            author = m.group(1).strip()
            title = m.group(2).strip()
            try:
                score = int(m.group(3))
                if 1 <= score <= 5:
                    scores[(author, title)] = score
            except ValueError:
                pass
        return scores

    def _match_poem_to_score(self, poem: Dict, scores: Dict[tuple, int]) -> Optional[int]:
        """Match poem to score by author/title (case-insensitive, fuzzy)."""
        author = poem.get('author', '')
        title = poem.get('title', '')
        key = (author, title)
        if key in scores:
            return scores[key]
        # Try case-insensitive
        key_lower = (author.lower(), title.lower())
        for (a, t), score in scores.items():
            if (a.lower(), t.lower()) == key_lower:
                return score
        # Try partial match
        for (a, t), score in scores.items():
            if author.lower() in a.lower() and title.lower() in t.lower():
                return score
            if a.lower() in author.lower() and t.lower() in title.lower():
                return score
        return None

    def parse_criterion_response(self, raw_response: str, poems: List[Dict], criterion: str) -> Dict:
        """Extract per-poem scores for one criterion from ranked list response."""
        scores = self._parse_ranked_list(raw_response)
        result = {}
        for poem in poems:
            score = self._match_poem_to_score(poem, scores)
            poem_id = poem.get('original_id', id(poem))
            result[poem_id] = score
        return result

    def annotate_batch_criterion(self, poems: List[Dict], criterion: str) -> tuple:
        """Run one criterion on a batch; returns (raw_response, parsed_scores dict)."""
        prompt = self.create_cat_prompt(poems, criterion)
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'top_p': 0.9,
                }
            )
            raw_text = response['response']
            scores = self.parse_criterion_response(raw_text, poems, criterion)
            return raw_text, scores
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return None, {}

    def annotate_batch(self, poems: List[Dict]) -> List[Dict]:
        """Run all criteria on batch, merge into one annotation per poem."""
        all_ratings = {p.get('original_id', id(p)): {} for p in poems}
        raw_responses = []

        rate_limit = self.config.get('ollama', {}).get('rate_limit_delay', 0.5)
        for criterion in self.criteria:
            raw, scores = self.annotate_batch_criterion(poems, criterion)
            raw_responses.append(f"=== {criterion} ===\n{raw or ''}")
            time.sleep(rate_limit)
            for poem_id, score in scores.items():
                if poem_id in all_ratings and score is not None:
                    all_ratings[poem_id][criterion] = score

        annotations = []
        for poem in poems:
            poem_id = poem.get('original_id', id(poem))
            ratings = all_ratings.get(poem_id, {})
            # Backward compatibility: map to old keys
            ratings_compat = {
                'creativity': ratings.get('creativity'),
                'quality': ratings.get('quality'),
                'innovative': ratings.get('innovativeness'),
                'similarity': ratings.get('similarity'),
                'poeticness': ratings.get('poeticness'),
            }
            annotations.append({
                'poem_id': poem_id,
                'poem': poem['text'],
                'metadata': {
                    'author': poem['author'],
                    'title': poem['title'],
                    'era': poem.get('era', 'unknown'),
                    'source': poem.get('source', ''),
                },
                'ratings': ratings_compat,
                'analysis': '\n\n'.join(raw_responses),
                'devices': 'n/a',
                'raw_response': '\n\n'.join(raw_responses),
                'model': self.model,
                'timestamp': time.time(),
            })
        return annotations

    def execute(self, input_data: Dict) -> Dict:
        """Main execution: load corpus, batch process, generate annotations."""
        corpus_file = input_data.get('corpus_file', 'data/processed/gpm_corpus.jsonl')
        output_file = Path('data/annotated/gpm_annotations.jsonl')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        poems = []
        with open(corpus_file) as f:
            for line in f:
                poems.append(json.loads(line))

        self.logger.info(f"Loaded {len(poems)} poems for annotation (batch_size={self.batch_size}, criteria={self.criteria})")

        state = self.load_state()
        start_idx = state.get('last_processed', 0) if state and state.get('status') != 'completed' else 0

        if start_idx > 0:
            annotations = []
            if output_file.exists():
                with open(output_file) as f:
                    for line in f:
                        annotations.append(json.loads(line))
        else:
            annotations = []
            output_file.write_text('')

        total_batches = (len(poems) - start_idx + self.batch_size - 1) // self.batch_size
        rate_limit = self.config.get('ollama', {}).get('rate_limit_delay', 0.5)

        for batch_num, i in enumerate(range(start_idx, len(poems), self.batch_size)):
            batch = poems[i:i + self.batch_size]
            batch_id = f"batch_{i // self.batch_size:04d}"

            self.logger.info(f"Processing {batch_id} ({batch_num + 1}/{total_batches})...")

            batch_annotations = self.annotate_batch(batch)
            annotations.extend(batch_annotations)

            with open(output_file, 'a') as f:
                for ann in batch_annotations:
                    ann['batch_id'] = batch_id
                    f.write(json.dumps(ann) + '\n')

            self.save_state({
                'last_processed': i + len(batch),
                'total_poems': len(poems),
                'completed_batches': batch_num + 1,
                'output_file': str(output_file)
            })

            time.sleep(rate_limit)

        successful = len([a for a in annotations if 'error' not in a])
        self.save_state({
            'status': 'completed',
            'total_annotated': len(annotations),
            'successful': successful,
            'failed': len(annotations) - successful,
            'output_file': str(output_file)
        })

        return {
            'annotations_file': str(output_file),
            'total_annotated': len(annotations),
            'success_rate': successful / len(annotations) if annotations else 0
        }
