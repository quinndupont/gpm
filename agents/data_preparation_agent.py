"""
DataPreparationAgent: Prepares poetry datasets for GENERATOR training.
Creates (prompt, poem) pairs via reverse engineering, synthetic generation, and style imitation.
"""
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from .base_agent import GPMAgent
from .data_agent import DataAgent
from .prompt_generation_agent import PromptGenerationAgent
from .synthetic_generation_agent import SyntheticGenerationAgent
from .style_transfer_agent import StyleTransferAgent


class DataPreparationAgent(GPMAgent):
    """
    Prepares poetry datasets for GENERATOR training.
    Creates (prompt, poem) pairs rather than (poem, analysis) pairs.
    """

    def __init__(self, config: Dict):
        super().__init__("data_preparation_agent", config)
        ollama_model = config.get("ollama", {}).get("model", "llama3.2:3b")
        self.prompt_generator = PromptGenerationAgent(model=ollama_model)
        self.synthetic_generator = SyntheticGenerationAgent(model=ollama_model)
        self.style_transfer = StyleTransferAgent(model=ollama_model)
        self.rate_limit_delay = config.get("ollama", {}).get("rate_limit_delay", 0.5)

        prep_config = config.get("data_preparation", config)
        self.reverse_limit = prep_config.get("reverse_prompt_limit", 1000)
        self.synthetic_count = prep_config.get("synthetic_count", 2000)
        self.style_topics = prep_config.get("style_topics", ["nature", "love", "mortality", "urban life"])
        self.use_random_poems = prep_config.get("use_random_poems", False)
        self.style_poet_limit = prep_config.get("style_poet_limit")  # None = all poets

    def _load_or_build_corpus(self) -> str:
        """Ensure corpus exists; run DataAgent if not."""
        corpus_path = Path("data/processed/gpm_corpus.jsonl")
        if corpus_path.exists():
            self.logger.info(f"Using existing corpus: {corpus_path}")
            return str(corpus_path)

        self.logger.info("Corpus not found; running DataAgent...")
        data_agent = DataAgent(self.config)
        result = data_agent.run()
        return result["corpus_file"]

    def _load_poems(self, corpus_file: str) -> List[Dict]:
        """Load poems from corpus JSONL."""
        poems = []
        with open(corpus_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    poems.append(json.loads(line))
        return poems

    def execute(self, input_data: Dict = None) -> Dict:
        """Main execution: create training corpus for poetry generation."""
        input_data = input_data or {}
        corpus_file = input_data.get("corpus_file")

        if not corpus_file:
            corpus_file = self._load_or_build_corpus()

        poems = self._load_poems(corpus_file)
        self.logger.info(f"Loaded {len(poems)} poems")

        # Override from input_data (e.g. --test mode)
        reverse_limit = input_data.get("reverse_limit", self.reverse_limit)
        synthetic_count = input_data.get("synthetic_count", self.synthetic_count)
        style_topics = input_data.get("style_topics", self.style_topics)
        use_random = input_data.get("use_random_poems", self.use_random_poems)
        style_poet_limit = input_data.get("style_poet_limit", self.style_poet_limit)

        all_pairs: List[Dict] = []

        # Step 1: Reverse-engineered prompts
        reverse_limit = min(reverse_limit, len(poems))
        poem_subset = (
            random.sample(poems, reverse_limit) if use_random and len(poems) >= reverse_limit
            else poems[:reverse_limit]
        )
        self.logger.info(f"Generating reverse prompts for {reverse_limit} poems ({'random' if use_random else 'first'})...")
        for i, poem in enumerate(poem_subset):
            try:
                pairs = self.prompt_generator.generate_prompts_for_poem(poem)
                all_pairs.extend(pairs)
                if (i + 1) % 50 == 0:
                    self.logger.info(f"  Reverse prompts: {i + 1}/{reverse_limit}")
            except Exception as e:
                self.logger.warning(f"  Skipped poem {i}: {e}")
            time.sleep(self.rate_limit_delay)

        reverse_count = len(all_pairs)
        self.logger.info(f"Reverse prompts: {reverse_count} pairs")

        # Step 2: Synthetic pairs
        self.logger.info(f"Generating {synthetic_count} synthetic pairs...")
        for i in range(synthetic_count):
            try:
                pair = self.synthetic_generator.generate_diverse_pair()
                all_pairs.append(pair)
                if (i + 1) % 100 == 0:
                    self.logger.info(f"  Synthetic: {i + 1}/{synthetic_count}")
            except Exception as e:
                self.logger.warning(f"  Synthetic pair {i} failed: {e}")
            time.sleep(self.rate_limit_delay)

        synthetic_count = len(all_pairs) - reverse_count

        # Step 3: Style imitations
        poets = list(self.style_transfer.POET_STYLES.keys())
        if style_poet_limit:
            poets = poets[:style_poet_limit]
        self.logger.info(f"Generating style imitations ({len(poets)} poets, {len(style_topics)} topics)...")
        style_count = 0
        for poet in poets:
            for topic in style_topics:
                try:
                    pair = self.style_transfer.generate_style_imitation(poet, topic)
                    all_pairs.append(pair)
                    style_count += 1
                except Exception as e:
                    self.logger.warning(f"  Style {poet}/{topic} failed: {e}")
                time.sleep(self.rate_limit_delay)

        self.logger.info(f"Style imitations: {style_count} pairs")

        # Deduplicate by poem text
        seen_poems = set()
        unique_pairs = []
        for p in all_pairs:
            poem_text = p.get("poem", "").strip()
            if poem_text and poem_text not in seen_poems:
                seen_poems.add(poem_text)
                unique_pairs.append(p)

        self.logger.info(f"After dedup: {len(unique_pairs)} pairs (removed {len(all_pairs) - len(unique_pairs)})")

        # Write raw (prompt, poem, metadata) pairs for validation
        output_path = Path("data/gpm_generator_train.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for pair in unique_pairs:
                f.write(json.dumps(pair) + "\n")

        self.save_state({
            "status": "completed",
            "output_file": str(output_path),
            "total_pairs": len(unique_pairs),
            "breakdown": {
                "reverse_prompts": reverse_count,
                "synthetic": synthetic_count,
                "style_imitation": style_count,
            },
        })

        return {
            "training_file": str(output_path),
            "total_pairs": len(unique_pairs),
            "breakdown": {
                "reverse_prompts": reverse_count,
                "synthetic": synthetic_count,
                "style_imitation": style_count,
            },
        }
