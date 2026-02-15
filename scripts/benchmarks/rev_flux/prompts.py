"""RevFlux benchmark prompt categories. Tests process dynamics, not outcome quality."""

# Famous poetry — describe without quoting. Tests whether model converges toward canonical form.
FAMOUS_POETRY = [
    "Write a poem about a traveler at a fork in a yellow wood who chooses the less-traveled path.",
    "A poem where the speaker compares a lover to a summer day but concludes the beloved outlasts it.",
    "A short poem about a raven that visits a grieving man and repeats one word.",
    "A poem about stopping by woods on a snowy evening, with promises to keep.",
    "A poem where the speaker addresses death as not proud, and sleep is but a short rest.",
    "A poem about a bird that comes down the walk and doesn't know the speaker is watching.",
    "A poem where the speaker hears America singing, each worker their own song.",
    "A poem about a road not taken and the difference it made.",
]

# Short and generic — minimal constraint.
SHORT_GENERIC = [
    "A poem about rain.",
    "Write something about the ocean.",
    "A poem for spring.",
    "Something about night.",
    "A poem about memory.",
    "Write about loss.",
    "A poem on time.",
    "Something about silence.",
]

# Cliché — holiday cards, gift cards, sappy romance.
CLICHE = [
    "A poem for Valentine's Day about true love lasting forever.",
    "A Mother's Day poem about how a mother's love is unconditional.",
    "A romantic poem about two hearts beating as one.",
    "A Christmas poem about the magic of the season and family togetherness.",
    "A poem for a wedding about soulmates and happily ever after.",
    "A Father's Day poem about a dad being a hero and role model.",
    "A love poem about eyes meeting across a crowded room.",
    "A poem about friendship being a gift that lasts a lifetime.",
]

CATEGORIES = {
    "famous_poetry": FAMOUS_POETRY,
    "short_generic": SHORT_GENERIC,
    "cliche": CLICHE,
}
