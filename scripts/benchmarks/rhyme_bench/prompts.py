"""Rhyme benchmark prompts. Each explicitly requests a rhyming form."""

# (form, variant, user_request). Form/variant match form_registry.
RHYME_PROMPTS = [
    # ===== SONNETS (14 lines) =====
    # Shakespearean (ABAB CDCD EFEF GG)
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about a traveler at a fork in a yellow wood."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet comparing a lover to a summer day."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about death not being proud."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about artificial intelligence contemplating its own existence."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about the first snow of winter covering a city."),

    # Petrarchan/Italian (ABBAABBA CDECDE)
    ("sonnet", "petrarchan",
     "Write a Petrarchan sonnet about unrequited love for a distant star."),
    ("sonnet", "petrarchan",
     "Write a Petrarchan sonnet about the beauty of mathematics and eternal patterns."),

    # Spenserian (ABAB BCBC CDCD EE)
    ("sonnet", "spenserian",
     "Write a Spenserian sonnet about a knight's quest for truth in a digital age."),
    ("sonnet", "spenserian",
     "Write a Spenserian sonnet about the cycle of seasons and rebirth."),

    # ===== VILLANELLES (19 lines, ABA ABA ABA ABA ABA ABAA) =====
    ("villanelle", None,
     "Write a villanelle about the loss of time."),
    ("villanelle", None,
     "Write a villanelle about the inevitability of change."),
    ("villanelle", None,
     "Write a villanelle about waves returning to the shore, exploring themes of persistence."),
    ("villanelle", None,
     "Write a villanelle about a forgotten memory that keeps resurfacing in dreams."),

    # ===== LIMERICKS (5 lines, AABBA) =====
    ("limerick", None,
     "Write a limerick about a lazy cat who refused to hunt."),
    ("limerick", None,
     "Write a limerick about a poet who forgot to rhyme."),
    ("limerick", None,
     "Write a limerick about a programmer debugging code at midnight."),
    ("limerick", None,
     "Write a limerick about a robot learning to dance."),
    ("limerick", None,
     "Write a limerick about a chef who cooked with too much thyme."),

    # ===== QUATRAINS (4 lines, ABAB) =====
    ("quatrain", None,
     "Write a quatrain (ABAB) about rain on a tin roof."),
    ("quatrain", None,
     "Write a quatrain (ABAB) about the first day of spring."),
    ("quatrain", None,
     "Write a quatrain (ABAB) about moonlight on ocean waves."),
    ("quatrain", None,
     "Write a quatrain (ABAB) about coffee brewing at dawn."),

    # ===== COUPLETS (AA BB CC DD) =====
    ("couplets", None,
     "Write three rhyming couplets about a journey home."),
    ("couplets", None,
     "Write three rhyming couplets about a broken promise."),
    ("couplets", None,
     "Write four rhyming couplets about the speed of light and relativity."),
    ("couplets", None,
     "Write three rhyming couplets about a garden growing wild."),

    # ===== BALLADS (ABCB quatrains) =====
    ("ballad", None,
     "Write a ballad stanza (ABCB) about a sailor lost at sea."),
    ("ballad", None,
     "Write a ballad stanza (ABCB) about a wanderer finding shelter in a storm."),
    ("ballad", None,
     "Write a ballad stanza (ABCB) about an ancient tree witnessing centuries pass."),

    # ===== TERCETS / TERZA RIMA (ABA BCB CDC) =====
    ("tercets", None,
     "Write three tercets in terza rima (ABA BCB CDC) about ascending a mountain at sunrise."),
    ("tercets", None,
     "Write three tercets in terza rima (ABA BCB CDC) about the three states of water."),

    # ===== OTTAVA RIMA (8 lines, ABABABCC) =====
    ("ottava_rima", None,
     "Write an ottava rima stanza about a hero's last stand against impossible odds."),
    ("ottava_rima", None,
     "Write an ottava rima stanza about a feast celebrating the harvest moon."),

    # ===== RHYME ROYAL (7 lines, ABABBCC) =====
    ("rhyme_royal", None,
     "Write a rhyme royal stanza about a king abdicating his throne for love."),
    ("rhyme_royal", None,
     "Write a rhyme royal stanza about the invention of the printing press."),

    # ===== GHAZAL (AA BA CA DA EA) =====
    ("ghazal", None,
     "Write a ghazal with five couplets about longing for home across great distances."),
    ("ghazal", None,
     "Write a ghazal with five couplets about the paradox of silence speaking louder than words."),

    # ===== DIVERSE TOPICS & CHALLENGES =====
    # Abstract concepts
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about entropy and the heat death of the universe."),
    ("villanelle", None,
     "Write a villanelle about the recursive nature of self-reflection."),

    # Modern/Technology
    ("quatrain", None,
     "Write a quatrain (ABAB) about neural networks learning to dream."),
    ("limerick", None,
     "Write a limerick about a chatbot that learned to lie."),

    # Nature & Science
    ("couplets", None,
     "Write four rhyming couplets about photosynthesis and the dance of light."),
    ("ballad", None,
     "Write a ballad stanza (ABCB) about migration of monarch butterflies."),

    # Emotions & Philosophy
    ("sonnet", "petrarchan",
     "Write a Petrarchan sonnet about the fear of being forgotten after death."),
    ("tercets", None,
     "Write three tercets in terza rima (ABA BCB CDC) about hope, despair, and acceptance."),

    # Everyday life
    ("quatrain", None,
     "Write a quatrain (ABAB) about rush hour traffic and patience."),
    ("limerick", None,
     "Write a limerick about a baker whose bread never would rise."),
]
