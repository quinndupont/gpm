import json
import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from nlp_dedup import Deduper

from .base_agent import GPMAgent


class DataAgent(GPMAgent):
    """
    Responsible for:
    1. Downloading poetry datasets from HuggingFace
    2. Normalizing schema across sources
    3. Cleaning text (Gutenberg headers, whitespace)
    4. Deduplication
    5. Quality filtering
    """
    DATASET_CONFIGS = {
        'gutenberg': {
            'repo': 'matthh/gutenberg-poetry-corpus',
            'text_field': 'content',
            'author_field': 'author',
            'title_field': 'title',
            'era': 'classic'
        },
        'merve': {
            'repo': 'merve/poetry',
            'text_field': 'content',
            'author_field': 'author',
            'title_field': 'poem name',
            'era_field': 'age',
            'type_field': 'type'
        },
        'public_domain': {
            'repo': 'DanFosing/public-domain-poetry',
            'text_field': 'text',
            'author_field': 'Author',
            'title_field': 'Title',
            'era': 'mixed'
        },
        'modern': {
            'repo': 'jassiyu/poetry-modern',
            'text_field': 'content',
            'author_field': 'author',
            'title_field': 'poem name',
            'era_field': 'age',
            'type_field': 'type',
            'era': 'contemporary'
        }
    }

    def __init__(self, config: Dict):
        super().__init__('data_agent', config)
        self.raw_dir = Path('data/raw')
        self.processed_dir = Path('data/processed')
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Aggressive cleaning while preserving poetic structure"""
        if not text or not isinstance(text, str):
            return ""

        text = re.sub(r'\*\*\* START OF .* \*\*\*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\*\*\* END OF .* \*\*\*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Produced by .*?(\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'End of (the )?Project Gutenberg.*', '', text, flags=re.DOTALL | re.IGNORECASE)

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            line = re.sub(r'\s+', ' ', line)
            if line:
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()

    def quality_filter(self, poem: Dict) -> bool:
        """Filter low-quality entries"""
        text = poem.get('text', '')

        if len(text) < 150 or len(text) > 8000:
            return False

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 4:
            return False

        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len > 150:
            return False

        if len(re.sub(r'[^a-zA-Z]', '', text)) < 100:
            return False

        return True

    def _create_deduper(self) -> Deduper:
        """Create Deduper with config from processing.dedup or defaults."""
        dedup_config = self.config.get('processing', {}).get('dedup', {})
        return Deduper(
            split_method=dedup_config.get('split_method', 'word_ngram'),
            ngram_size=dedup_config.get('ngram_size', 13),
            similarity_threshold=dedup_config.get('similarity_threshold', 0.8),
            num_minhashes=dedup_config.get('num_minhashes', 128),
            store_corpus_to_disk=False,
            store_mask_to_disk=False,
            store_lsh_cache_to_disk=False,
            store_config_to_disk=False,
            return_generator=True,
            verbose=False,
        )

    def deduplicate(self, poems: List[Dict]) -> List[Dict]:
        """Remove near-duplicates using nlp_dedup (MinHash)."""
        if not poems:
            return poems

        deduper = self._create_deduper()
        mask_gen = deduper.deduplicate(
            corpus=poems,
            text_column='text',
            output_dir='checkpoints/nlp_dedup_tmp',
            overwrite=True,
            num_docs=len(poems),
        )
        mask = list(mask_gen)
        return [poems[i] for i, m in enumerate(mask) if not m['duplicate']]

    def normalize_schema(self, dataset_name: str, raw_data) -> List[Dict]:
        """Convert all datasets to common schema"""
        config = self.DATASET_CONFIGS[dataset_name]
        poems = []
        text_field = config['text_field']

        for item in raw_data:
            text = item.get(text_field) or item.get(text_field.capitalize()) or ''
            cleaned_text = self.clean_text(text)

            if not cleaned_text:
                continue

            def get_val(field_key, default=''):
                key = config.get(field_key, field_key)
                v = item.get(key) or (item.get(key.capitalize()) if isinstance(key, str) else None)
                return v if v is not None else default

            poem = {
                'text': cleaned_text,
                'author': str(get_val('author_field', 'Unknown')),
                'title': str(get_val('title_field', 'Untitled')),
                'source': dataset_name,
                'era': get_val('era_field') or config.get('era', 'unknown'),
                'tags': [dataset_name],
                'original_id': hashlib.md5(text[:100].encode()).hexdigest()[:12]
            }

            if 'type_field' in config:
                poem['type'] = get_val('type_field', 'general')
                poem['tags'].append(poem['type'])

            poems.append(poem)

        return poems

    def _load_local_downloads(self) -> List[Dict]:
        """Load curated .txt files from data/downloads/ as corpus poems."""
        downloads_dir = Path('data/downloads')
        if not downloads_dir.exists():
            return []

        txt_files = sorted(downloads_dir.glob('*.txt'))
        poems = []
        for txt_file in txt_files:
            if 'readme' in txt_file.name.lower():
                continue
            try:
                raw_text = txt_file.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue

            # Derive author from filename (e.g. "blake_songs_innocence_experience.txt")
            stem = txt_file.stem
            parts = stem.split('_', 1)
            author = parts[0].title() if parts else 'Unknown'
            title = parts[1].replace('_', ' ').title() if len(parts) > 1 else stem

            cleaned = self.clean_text(raw_text)
            if not cleaned:
                continue

            poem = {
                'text': cleaned,
                'author': author,
                'title': title,
                'source': 'curated_downloads',
                'era': 'classic',
                'tags': ['curated_downloads'],
                'original_id': hashlib.md5(cleaned[:100].encode()).hexdigest()[:12],
            }

            if self.quality_filter(poem):
                poems.append(poem)

        return poems

    def _load_poems(self, dataset_name: str, config: Dict) -> List[Dict]:
        """Load poems from local processed file or HuggingFace."""
        proc_config = self.config.get('processing', {})
        local_file = self.processed_dir / f"{dataset_name}_cleaned.jsonl"

        if proc_config.get('use_local_processed') and local_file.exists():
            self.logger.info(f"  Loading from local cache: {local_file}")
            poems = []
            with open(local_file) as f:
                for line in f:
                    poems.append(json.loads(line))
            return poems

        if proc_config.get('use_offline'):
            os.environ['HF_HUB_OFFLINE'] = '1'

        try:
            ds = load_dataset(config['repo'], split='train')
            self.logger.info(f"  Downloaded {len(ds)} rows")
        except Exception as e:
            self.logger.error(f"  Failed to load {dataset_name}: {e}")
            raise

        poems = self.normalize_schema(dataset_name, ds)
        self.logger.info(f"  Normalized to {len(poems)} poems")
        poems = [p for p in poems if self.quality_filter(p)]
        self.logger.info(f"  After quality filter: {len(poems)}")
        poems = self.deduplicate(poems)
        self.logger.info(f"  After deduplication: {len(poems)}")

        with open(local_file, 'w') as f:
            for p in poems:
                f.write(json.dumps(p) + '\n')
        return poems

    def execute(self, input_data: Dict = None) -> Dict:
        """Main execution: download, clean, dedupe, save"""
        all_poems = []
        stats = {}

        for dataset_name, config in self.DATASET_CONFIGS.items():
            self.logger.info(f"Processing {dataset_name}...")
            try:
                poems = self._load_poems(dataset_name, config)
            except Exception:
                continue

            all_poems.extend(poems)
            stats[dataset_name] = len(poems)

        # Load curated downloads from data/downloads/
        self.logger.info("Processing curated downloads...")
        local_poems = self._load_local_downloads()
        if local_poems:
            all_poems.extend(local_poems)
            stats['curated_downloads'] = len(local_poems)
            self.logger.info(f"  Added {len(local_poems)} poems from data/downloads/")

        self.logger.info(f"Global deduplication: {len(all_poems)} total...")
        all_poems = self.deduplicate(all_poems)
        self.logger.info(f"After global dedup: {len(all_poems)}")

        final_file = self.processed_dir / 'gpm_corpus.jsonl'
        with open(final_file, 'w') as f:
            for p in all_poems:
                f.write(json.dumps(p) + '\n')

        stats['total_unique'] = len(all_poems)
        self.save_state({
            'status': 'completed',
            'stats': stats,
            'output_file': str(final_file)
        })

        return {
            'corpus_file': str(final_file),
            'total_poems': len(all_poems),
            'stats': stats
        }
