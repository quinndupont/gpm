"""Rhyme benchmark prompts. Each explicitly requests a rhyming form."""

# (form, variant, user_request). Form/variant match form_registry.
RHYME_PROMPTS = [
    # Sonnet (Shakespearean: ABAB CDCD EFEF GG)
    ("sonnet", "shakespearean", "Write a Shakespearean sonnet about a traveler at a fork in a yellow wood."),
    ("sonnet", "shakespearean", "Write a Shakespearean sonnet comparing a lover to a summer day."),
    ("sonnet", "shakespearean", "Write a Shakespearean sonnet about death not being proud."),
    # Villanelle (ABA ABA ABA ABA ABA ABAA)
    ("villanelle", None, "Write a villanelle about the loss of time."),
    ("villanelle", None, "Write a villanelle about the inevitability of change."),
    # Limerick (AABBA)
    ("limerick", None, "Write a limerick about a lazy cat who refused to hunt."),
    ("limerick", None, "Write a limerick about a poet who forgot to rhyme."),
    # Quatrain (ABAB)
    ("quatrain", None, "Write a quatrain (ABAB) about rain on a tin roof."),
    ("quatrain", None, "Write a quatrain (ABAB) about the first day of spring."),
    # Couplets (AA BB CC)
    ("couplets", None, "Write three rhyming couplets about a journey home."),
    ("couplets", None, "Write three rhyming couplets about a broken promise."),
]
