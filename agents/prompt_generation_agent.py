"""
PromptGenerationAgent: Reverse-engineers writing prompts from existing poems.
Creates (prompt, poem) pairs for poetry generator training.
"""
import re
from typing import List, Dict

import ollama


class PromptGenerationAgent:
    """
    Generates diverse writing prompts for existing poems.
    This creates (prompt, poem) pairs for training.
    """

    REVERSE_PROMPT_TEMPLATE = """You are a creative writing instructor.
Given this poem, generate 3-5 different prompts that could have inspired it.

POEM:
"{title}" by {author}

{poem_text}

Generate prompts in these categories:

1. THEMATIC: Focus on subject matter (love, nature, mortality, etc.)
2. FORMAL: Specify structure (sonnet, haiku, free verse, etc.)
3. STYLISTIC: Mimic a poet or style (like Dickinson, like spoken word)
4. CONSTRAINT-BASED: Include specific requirements (must use "X" word, no rhyme, etc.)
5. EMOTIONAL: Focus on mood/feeling to evoke

FORMAT:
Prompt 1 [Thematic]: "Write a poem about..."
Prompt 2 [Formal]: "Compose a sonnet that..."
etc.

Make prompts specific and evocative. Avoid generic instructions."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    def generate_prompts_for_poem(self, poem: Dict) -> List[Dict]:
        """Generate multiple prompts for a single poem"""
        full_prompt = self.REVERSE_PROMPT_TEMPLATE.format(
            title=poem.get("title", "Untitled"),
            author=poem.get("author", "Unknown"),
            poem_text=poem.get("text", ""),
        )

        response = ollama.generate(
            model=self.model,
            prompt=full_prompt,
            options={"temperature": 0.9, "num_predict": 800},
        )

        prompts = self._parse_prompts(response["response"])

        return [
            {
                "prompt": p["text"],
                "prompt_type": p["type"],
                "poem": poem["text"],
                "metadata": {
                    "author": poem.get("author"),
                    "title": poem.get("title"),
                    "source": poem.get("source"),
                    "generation_method": "reverse_engineered",
                },
            }
            for p in prompts
        ]

    def _parse_prompts(self, text: str) -> List[Dict]:
        """Parse prompt categories from model output"""
        prompts = []
        pattern = r'(?:Prompt\s*\d+\s*)?\[([^\]]+)\]:\s*"([^"]+)"'
        matches = re.findall(pattern, text)

        for ptype, ptext in matches:
            prompts.append({"type": ptype.strip(), "text": ptext.strip()})

        return prompts if prompts else [{"type": "general", "text": text.strip()[:500]}]
