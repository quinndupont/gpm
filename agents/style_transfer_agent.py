"""
StyleTransferAgent: Generates poems in specific poet styles.
Based on the "imitation" method from generative aesthetics research.
"""
from typing import Dict

import ollama


class StyleTransferAgent:
    """
    Generate poems in specific poet styles.
    """

    STYLE_IMITATION_PROMPT = """Write an original poem in the style of {poet}.

Style characteristics to emulate:
{style_description}

TOPIC: {topic}

Write an original poem (do NOT copy existing poems by {poet}).
Capture their distinctive voice, vocabulary, and approach, but with entirely new content.
Write only the poem, no commentary."""

    POET_STYLES = {
        "Emily Dickinson": "short lines, dashes for pauses, slant rhyme, domestic imagery, metaphysical themes, capitalization of nouns",
        "Walt Whitman": "long lines, free verse, catalogs/lists, democratic inclusiveness, cosmic vision, first-person plural",
        "William Carlos Williams": "imagist precision, everyday objects, variable foot, American vernacular, visual clarity",
        "Sylvia Plath": "intense imagery, confessional tone, harsh consonants, themes of death and rebirth, domestic surrealism",
        "Langston Hughes": "jazz rhythms, blues structure, African American vernacular, social commentary, musicality",
        "Mary Oliver": "nature observation, accessible language, spiritual wonder, precise animal/plant imagery, gentle tone",
    }

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    def generate_style_imitation(self, poet: str, topic: str) -> Dict:
        """Generate poem imitating specific poet's style"""
        style_desc = self.POET_STYLES.get(poet, "distinctive poetic voice")

        prompt = self.STYLE_IMITATION_PROMPT.format(
            poet=poet,
            style_description=style_desc,
            topic=topic,
        )

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.85, "num_predict": 600},
        )

        return {
            "prompt": f"Write a poem in the style of {poet} about {topic}",
            "poem": response["response"].strip(),
            "metadata": {
                "target_poet": poet,
                "topic": topic,
                "style_description": style_desc,
                "generation_method": "style_imitation",
            },
        }
