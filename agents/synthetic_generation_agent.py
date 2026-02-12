"""
SyntheticGenerationAgent: Generates entirely new (prompt, poem) pairs using local LLM.
Based on NVIDIA NeMo synthetic data generation patterns.
"""
import random
from typing import Dict

import ollama


class SyntheticGenerationAgent:
    """
    Generate entirely new (prompt, poem) pairs using local LLM.
    """

    DIVERSE_PROMPT_TEMPLATE = """Generate a creative writing prompt for a poem.

Requirements:
- Topic: {topic}
- Style: {style}
- Form: {form}
- Constraint: {constraint}

Make the prompt specific and inspiring. Write only the prompt, no explanation."""

    POEM_GENERATION_TEMPLATE = """You are a gifted poet. Write an original poem based on this prompt:

{prompt}

Requirements:
- Original imagery (avoid clichés)
- Strong emotional resonance
- Attention to sound and rhythm
- {form_specific_instructions}

Write only the poem, no commentary."""

    TOPICS = [
        "autumn and memory",
        "urban isolation",
        "coastal landscapes",
        "family history",
        "technological alienation",
        "mythological retelling",
        "domestic rituals",
        "extinction and loss",
        "first love",
        "aging",
    ]

    STYLES = [
        "imagist (precise, clear images)",
        "confessional (personal, raw)",
        "nature romanticism",
        "modernist fragmentation",
        "spoken word rhythm",
        "surrealist (dream logic)",
        "minimalist (sparse, essential)",
    ]

    FORMS = [
        "free verse",
        "sonnet (Shakespearean)",
        "sonnet (Petrarchan)",
        "haiku sequence",
        "villanelle",
        "blank verse",
        "concrete poetry",
        "prose poem",
        "ghazal",
        "sestina",
    ]

    CONSTRAINTS = [
        "must include the color blue",
        "no adjectives allowed",
        "every line must start with a verb",
        "include a childhood memory",
        "use scientific terminology metaphorically",
        "avoid the word 'love'",
        "write from the perspective of an object",
    ]

    FORM_INSTRUCTIONS = {
        "sonnet (Shakespearean)": "14 lines, ABAB CDCD EFEF GG rhyme scheme, iambic pentameter, final couplet resolution",
        "sonnet (Petrarchan)": "14 lines, ABBA ABBA CDECDE or CDCDCD, octave presents problem, sestet resolves",
        "villanelle": "19 lines, 5 tercets + 1 quatrain, ABA rhyme, refrains at lines 1, 6, 12, 18 and 3, 9, 15, 19",
        "haiku sequence": "3 haikus (5-7-5 syllables each), connected by theme, present-tense imagery",
        "sestina": "39 lines, 6 stanzas of 6 lines + envoi, end-words rotate: ABCDEF, FAEBDC, CFDABE, ECBFAD, DEACFB, BDFECA",
        "free verse": "no set meter or rhyme, but maintain rhythm through line breaks and cadence",
    }

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    def generate_diverse_pair(self) -> Dict:
        """Generate one synthetic (prompt, poem) pair with random parameters"""
        topic = random.choice(self.TOPICS)
        style = random.choice(self.STYLES)
        form = random.choice(self.FORMS)
        constraint = random.choice(self.CONSTRAINTS)

        prompt_gen = self.DIVERSE_PROMPT_TEMPLATE.format(
            topic=topic,
            style=style,
            form=form,
            constraint=constraint,
        )

        prompt_response = ollama.generate(
            model=self.model,
            prompt=prompt_gen,
            options={"temperature": 0.9, "num_predict": 200},
        )
        prompt_text = prompt_response["response"].strip()

        form_instructions = self._get_form_instructions(form)
        poem_gen = self.POEM_GENERATION_TEMPLATE.format(
            prompt=prompt_text,
            form_specific_instructions=form_instructions,
        )

        poem_response = ollama.generate(
            model=self.model,
            prompt=poem_gen,
            options={"temperature": 0.8, "num_predict": 600},
        )
        poem_text = poem_response["response"].strip()

        return {
            "prompt": prompt_text,
            "poem": poem_text,
            "metadata": {
                "topic": topic,
                "style": style,
                "form": form,
                "constraint": constraint,
                "generation_method": "synthetic_full",
            },
        }

    def _get_form_instructions(self, form: str) -> str:
        """Get specific instructions for poetic forms"""
        return self.FORM_INSTRUCTIONS.get(
            form, "follow the conventions of this form precisely"
        )
