"""Form registry — maps poetic form names to expected rhyme schemes and constraints."""
import re

FORMS = {
    "sonnet": {
        "rhyme_scheme": "ABAB CDCD EFEF GG",
        "line_count": 14,
        "variants": {
            "petrarchan": {"rhyme_scheme": "ABBAABBA CDECDE", "line_count": 14},
            "italian": {"rhyme_scheme": "ABBAABBA CDECDE", "line_count": 14},
            "spenserian": {"rhyme_scheme": "ABAB BCBC CDCD EE", "line_count": 14},
            "shakespearean": {"rhyme_scheme": "ABAB CDCD EFEF GG", "line_count": 14},
        },
    },
    "villanelle": {
        "rhyme_scheme": "ABA ABA ABA ABA ABA ABAA",
        "line_count": 19,
        "refrains": [1, 3],
    },
    "limerick": {
        "rhyme_scheme": "AABBA",
        "line_count": 5,
    },
    "ghazal": {
        "rhyme_scheme": "AA BA CA DA EA",
        "couplet_form": True,
    },
    "couplets": {
        "rhyme_scheme": "AA BB CC DD",
        "repeating_unit": "AA",
    },
    "tercets": {
        "rhyme_scheme": "ABA BCB CDC DED",
        "repeating_unit": "ABA",
        "aliases": ["terza rima"],
    },
    "ballad": {
        "rhyme_scheme": "ABCB",
        "quatrain_form": True,
        "aliases": ["ballad stanza"],
    },
    "quatrain": {
        "rhyme_scheme": "ABAB",
        "quatrain_form": True,
    },
    "ottava_rima": {
        "rhyme_scheme": "ABABABCC",
        "line_count": 8,
        "aliases": ["ottava rima"],
    },
    "rhyme_royal": {
        "rhyme_scheme": "ABABBCC",
        "line_count": 7,
        "aliases": ["rhyme royal"],
    },
    "free_verse": {
        "rhyme_scheme": None,
    },
}

# Forms where rhyming is expected
RHYMING_FORMS = {k for k, v in FORMS.items() if v.get("rhyme_scheme") is not None}

# Build keyword list for detection (includes aliases)
_FORM_KEYWORDS: list[tuple[str, str]] = []
for form_name, spec in FORMS.items():
    if form_name == "free_verse":
        continue
    # Add the canonical name
    _FORM_KEYWORDS.append((form_name.replace("_", " "), form_name))
    # Add aliases
    for alias in spec.get("aliases", []):
        _FORM_KEYWORDS.append((alias.lower(), form_name))
# Sort longest-first so "terza rima" matches before "rima"
_FORM_KEYWORDS.sort(key=lambda x: -len(x[0]))


def detect_form(text: str) -> str | None:
    """Check if text mentions a known poetic form. Returns canonical form name or None."""
    lower = text.lower()
    for keyword, form_name in _FORM_KEYWORDS:
        if keyword in lower:
            return form_name
    return None


def get_scheme(form: str, variant: str | None = None) -> str | None:
    """Get the expected rhyme scheme string for a form (and optional variant)."""
    spec = FORMS.get(form)
    if not spec:
        return None
    if variant and "variants" in spec:
        v = spec["variants"].get(variant)
        if v:
            return v.get("rhyme_scheme", spec.get("rhyme_scheme"))
    return spec.get("rhyme_scheme")


def parse_scheme(scheme_str: str) -> list[str]:
    """Parse a scheme string like 'ABAB CDCD EFEF GG' into a flat list of letters."""
    if not scheme_str:
        return []
    return list(scheme_str.replace(" ", ""))


def get_line_count(form: str, variant: str | None = None) -> int | None:
    """Get expected line count, or None if variable-length."""
    spec = FORMS.get(form)
    if not spec:
        return None
    if variant and "variants" in spec:
        v = spec["variants"].get(variant)
        if v and "line_count" in v:
            return v["line_count"]
    return spec.get("line_count")


def is_rhyming_form(form: str) -> bool:
    """Return True if this form expects rhyming."""
    return form in RHYMING_FORMS


def form_description(form: str, variant: str | None = None) -> str:
    """Human-readable description for use in briefs/prompts."""
    spec = FORMS.get(form)
    if not spec:
        return ""
    scheme = get_scheme(form, variant)
    lc = get_line_count(form, variant)
    name = variant or form.replace("_", " ")
    parts = [name.title()]
    if lc:
        parts.append(f"{lc} lines")
    if scheme:
        parts.append(f"rhyme scheme {scheme}")
    return " — ".join(parts)
