# GPM Test Suite

Tests are organized by concern. Run with `make test` (fast) or `make test-all`.

## Markers

| Marker | Description |
|--------|-------------|
| `data` | Data validation (schema, quality gate, chat format, rhyme density) |
| `prompts` | Prompt loader, template rendering, persona loading |
| `eval` | Rhyme analyzer, form registry, meter analyzer |
| `slow` | GPU training dry-run, full benchmarks (skipped in CI) |

## Running subsets

```bash
pytest tests/ -m data -v
pytest tests/ -m prompts -v
pytest tests/ -m eval -v
pytest tests/ -m "not slow" -v   # Default for CI
```

## Fixtures

`tests/fixtures/` contains committed sample data:

- `sample_critique.jsonl` — educator critiques
- `sample_pairs.jsonl` — brief → poem pairs
- `sample_rhyme_pairs.jsonl` — rhyme-form pairs
- `sample_train.jsonl` — chat-format training examples
- `sample_poem_good.txt`, `sample_poem_bad.txt` — poem samples

Data validation tests use these fixtures. Tests that require `data/` (e.g. rhyme density on `data/rhyme_training/train.jsonl`) skip when the file is absent.

## Adding tests

- **New data schema:** Add schema assertion in `test_data_validation.py` and a fixture in `tests/fixtures/`.
- **New prompt:** Add rendering test in `test_prompts.py`; include in parametrized `test_get_prompt_succeeds` if applicable.
- **New eval logic:** Add unit test in `test_eval.py` with a known poem and expected output.
