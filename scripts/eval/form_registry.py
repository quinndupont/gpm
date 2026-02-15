"""Form registry — maps poetic form names to expected rhyme schemes and constraints."""
import re

FORMS = {
    "sonnet": {
        "rhyme_scheme": "ABAB CDCD EFEF GG",
        "line_count": 14,
        "meter": "iambic pentameter",
        "variants": {
            "petrarchan": {"rhyme_scheme": "ABBAABBA CDECDE", "line_count": 14, "meter": "iambic pentameter"},
            "italian": {"rhyme_scheme": "ABBAABBA CDECDE", "line_count": 14, "meter": "iambic pentameter"},
            "spenserian": {"rhyme_scheme": "ABAB BCBC CDCD EE", "line_count": 14, "meter": "iambic pentameter"},
            "shakespearean": {"rhyme_scheme": "ABAB CDCD EFEF GG", "line_count": 14, "meter": "iambic pentameter"},
        },
    },
    "villanelle": {
        "rhyme_scheme": "ABA ABA ABA ABA ABA ABAA",
        "line_count": 19,
        "meter": "iambic pentameter",
        "refrains": [1, 3],
    },
    "limerick": {
        "rhyme_scheme": "AABBA",
        "line_count": 5,
        "meter": "anapestic",
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
        "meter": "iambic",
        "aliases": ["terza rima"],
    },
    "ballad": {
        "rhyme_scheme": "ABCB",
        "quatrain_form": True,
        "meter": "iambic",
        "aliases": ["ballad stanza"],
    },
    "quatrain": {
        "rhyme_scheme": "ABAB",
        "quatrain_form": True,
    },
    "ottava_rima": {
        "rhyme_scheme": "ABABABCC",
        "line_count": 8,
        "meter": "iambic pentameter",
        "aliases": ["ottava rima"],
    },
    "rhyme_royal": {
        "rhyme_scheme": "ABABBCC",
        "line_count": 7,
        "meter": "iambic pentameter",
        "aliases": ["rhyme royal"],
    },
    "free_verse": {
        "rhyme_scheme": None,
    },
}

# Meter definitions: foot pattern + expected feet per line
METERS = {
    "iambic pentameter": {"foot": "01", "feet_per_line": 5},
    "iambic tetrameter": {"foot": "01", "feet_per_line": 4},
    "iambic trimeter": {"foot": "01", "feet_per_line": 3},
    "iambic": {"foot": "01", "feet_per_line": None},  # any length
    "trochaic": {"foot": "10", "feet_per_line": None},
    "trochaic tetrameter": {"foot": "10", "feet_per_line": 4},
    "anapestic": {"foot": "001", "feet_per_line": None},
    "dactylic": {"foot": "100", "feet_per_line": None},
    "spondaic": {"foot": "11", "feet_per_line": None},
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


def get_meter(form: str, variant: str | None = None) -> str | None:
    """Get the expected meter name for a form (e.g. 'iambic pentameter')."""
    spec = FORMS.get(form)
    if not spec:
        return None
    if variant and "variants" in spec:
        v = spec["variants"].get(variant)
        if v and "meter" in v:
            return v["meter"]
    return spec.get("meter")


def get_meter_spec(meter_name: str) -> dict | None:
    """Get the foot pattern and feet-per-line for a named meter."""
    return METERS.get(meter_name)


def is_rhyming_form(form: str) -> bool:
    """Return True if this form expects rhyming."""
    return form in RHYMING_FORMS


def is_metered_form(form: str, variant: str | None = None) -> bool:
    """Return True if this form expects a specific meter."""
    return get_meter(form, variant) is not None


def form_description(form: str, variant: str | None = None) -> str:
    """Human-readable description for use in briefs/prompts."""
    spec = FORMS.get(form)
    if not spec:
        return ""
    scheme = get_scheme(form, variant)
    lc = get_line_count(form, variant)
    meter = get_meter(form, variant)
    name = variant or form.replace("_", " ")
    parts = [name.title()]
    if lc:
        parts.append(f"{lc} lines")
    if meter:
        parts.append(meter)
    if scheme:
        parts.append(f"rhyme scheme {scheme}")
    return " — ".join(parts)
