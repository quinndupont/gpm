"""Rhyme benchmark prompts. Each explicitly requests a rhyming form."""

# (form, variant, user_request). Form/variant match form_registry.
RHYME_PROMPTS = [
    # ===== SONNETS (14 lines) =====
    # Shakespearean (ABAB CDCD EFEF GG)
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about a traveler at a fork in a yellow wood. Use the exact rhyme scheme: ABAB CDCD EFEF GG."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet comparing a lover to a summer day. Use the exact rhyme scheme: ABAB CDCD EFEF GG."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about death not being proud. Use the exact rhyme scheme: ABAB CDCD EFEF GG."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about artificial intelligence contemplating its own existence. Use the exact rhyme scheme: ABAB CDCD EFEF GG."),
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about the first snow of winter covering a city. Use the exact rhyme scheme: ABAB CDCD EFEF GG."),

    # Petrarchan/Italian (ABBAABBA CDECDE)
    ("sonnet", "petrarchan",
     "Write a Petrarchan sonnet about unrequited love for a distant star. Use the exact rhyme scheme: ABBAABBA CDECDE."),
    ("sonnet", "petrarchan",
     "Write a Petrarchan sonnet about the beauty of mathematics and eternal patterns. Use the exact rhyme scheme: ABBAABBA CDECDE."),

    # Spenserian (ABAB BCBC CDCD EE)
    ("sonnet", "spenserian",
     "Write a Spenserian sonnet about a knight's quest for truth in a digital age. Use the exact rhyme scheme: ABAB BCBC CDCD EE."),
    ("sonnet", "spenserian",
     "Write a Spenserian sonnet about the cycle of seasons and rebirth. Use the exact rhyme scheme: ABAB BCBC CDCD EE."),

    # ===== VILLANELLES (19 lines, ABA ABA ABA ABA ABA ABAA) =====
    ("villanelle", None,
     "Write a villanelle about the loss of time. Use the exact rhyme scheme: ABA ABA ABA ABA ABA ABAA."),
    ("villanelle", None,
     "Write a villanelle about the inevitability of change. Use the exact rhyme scheme: ABA ABA ABA ABA ABA ABAA."),
    ("villanelle", None,
     "Write a villanelle about waves returning to the shore, exploring themes of persistence. Use the exact rhyme scheme: ABA ABA ABA ABA ABA ABAA."),
    ("villanelle", None,
     "Write a villanelle about a forgotten memory that keeps resurfacing in dreams. Use the exact rhyme scheme: ABA ABA ABA ABA ABA ABAA."),

    # ===== LIMERICKS (5 lines, AABBA) =====
    ("limerick", None,
     "Write a limerick about a lazy cat who refused to hunt. Use the exact rhyme scheme: AABBA."),
    ("limerick", None,
     "Write a limerick about a poet who forgot to rhyme. Use the exact rhyme scheme: AABBA."),
    ("limerick", None,
     "Write a limerick about a programmer debugging code at midnight. Use the exact rhyme scheme: AABBA."),
    ("limerick", None,
     "Write a limerick about a robot learning to dance. Use the exact rhyme scheme: AABBA."),
    ("limerick", None,
     "Write a limerick about a chef who cooked with too much thyme. Use the exact rhyme scheme: AABBA."),

    # ===== QUATRAINS (4 lines, ABAB) =====
    ("quatrain", None,
     "Write a quatrain about rain on a tin roof. Use the exact rhyme scheme: ABAB."),
    ("quatrain", None,
     "Write a quatrain about the first day of spring. Use the exact rhyme scheme: ABAB."),
    ("quatrain", None,
     "Write a quatrain about moonlight on ocean waves. Use the exact rhyme scheme: ABAB."),
    ("quatrain", None,
     "Write a quatrain about coffee brewing at dawn. Use the exact rhyme scheme: ABAB."),

    # ===== COUPLETS (AA BB CC DD) =====
    ("couplets", None,
     "Write three rhyming couplets about a journey home. Use the exact rhyme scheme: AA BB CC."),
    ("couplets", None,
     "Write three rhyming couplets about a broken promise. Use the exact rhyme scheme: AA BB CC."),
    ("couplets", None,
     "Write four rhyming couplets about the speed of light and relativity. Use the exact rhyme scheme: AA BB CC DD."),
    ("couplets", None,
     "Write three rhyming couplets about a garden growing wild. Use the exact rhyme scheme: AA BB CC."),

    # ===== BALLADS (ABCB quatrains) =====
    ("ballad", None,
     "Write a ballad stanza about a sailor lost at sea. Use the exact rhyme scheme: ABCB."),
    ("ballad", None,
     "Write a ballad stanza about a wanderer finding shelter in a storm. Use the exact rhyme scheme: ABCB."),
    ("ballad", None,
     "Write a ballad stanza about an ancient tree witnessing centuries pass. Use the exact rhyme scheme: ABCB."),

    # ===== TERCETS / TERZA RIMA (ABA BCB CDC) =====
    ("tercets", None,
     "Write three tercets in terza rima about ascending a mountain at sunrise. Use the exact rhyme scheme: ABA BCB CDC."),
    ("tercets", None,
     "Write three tercets in terza rima about the three states of water. Use the exact rhyme scheme: ABA BCB CDC."),

    # ===== OTTAVA RIMA (8 lines, ABABABCC) =====
    ("ottava_rima", None,
     "Write an ottava rima stanza about a hero's last stand against impossible odds. Use the exact rhyme scheme: ABABABCC."),
    ("ottava_rima", None,
     "Write an ottava rima stanza about a feast celebrating the harvest moon. Use the exact rhyme scheme: ABABABCC."),

    # ===== RHYME ROYAL (7 lines, ABABBCC) =====
    ("rhyme_royal", None,
     "Write a rhyme royal stanza about a king abdicating his throne for love. Use the exact rhyme scheme: ABABBCC."),
    ("rhyme_royal", None,
     "Write a rhyme royal stanza about the invention of the printing press. Use the exact rhyme scheme: ABABBCC."),

    # ===== GHAZAL (AA BA CA DA EA) =====
    ("ghazal", None,
     "Write a ghazal with five couplets about longing for home across great distances. Use the exact rhyme scheme: AA BA CA DA EA."),
    ("ghazal", None,
     "Write a ghazal with five couplets about the paradox of silence speaking louder than words. Use the exact rhyme scheme: AA BA CA DA EA."),

    # ===== DIVERSE TOPICS & CHALLENGES =====
    # Abstract concepts
    ("sonnet", "shakespearean",
     "Write a Shakespearean sonnet about entropy and the heat death of the universe. Use the exact rhyme scheme: ABAB CDCD EFEF GG."),
    ("villanelle", None,
     "Write a villanelle about the recursive nature of self-reflection. Use the exact rhyme scheme: ABA ABA ABA ABA ABA ABAA."),

    # Modern/Technology
    ("quatrain", None,
     "Write a quatrain about neural networks learning to dream. Use the exact rhyme scheme: ABAB."),
    ("limerick", None,
     "Write a limerick about a chatbot that learned to lie. Use the exact rhyme scheme: AABBA."),

    # Nature & Science
    ("couplets", None,
     "Write four rhyming couplets about photosynthesis and the dance of light. Use the exact rhyme scheme: AA BB CC DD."),
    ("ballad", None,
     "Write a ballad stanza about migration of monarch butterflies. Use the exact rhyme scheme: ABCB."),

    # Emotions & Philosophy
    ("sonnet", "petrarchan",
     "Write a Petrarchan sonnet about the fear of being forgotten after death. Use the exact rhyme scheme: ABBAABBA CDECDE."),
    ("tercets", None,
     "Write three tercets in terza rima about hope, despair, and acceptance. Use the exact rhyme scheme: ABA BCB CDC."),

    # Everyday life
    ("quatrain", None,
     "Write a quatrain about rush hour traffic and patience. Use the exact rhyme scheme: ABAB."),
    ("limerick", None,
     "Write a limerick about a baker whose bread never would rise. Use the exact rhyme scheme: AABBA."),
]
