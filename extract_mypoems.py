#!/usr/bin/env python3
"""Extract poems from Cocoa HTML Writer exports in data/mypoems/ to JSONL."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path


class PoemExtractor(HTMLParser):
    """Parse Cocoa HTML Writer format: title from div.title, lines from p.p2 spans."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.date = ""
        self.lines = []
        self._in_title = False
        self._in_header = False
        self._in_poem_line = False
        self._current_line = ""
        self._current_p_class = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        if tag == "div" and "title" in cls.split():
            self._in_title = True
        elif tag == "div" and "pageHeader" in cls:
            self._in_header = True
        elif tag == "p":
            self._current_p_class = cls
            if "p2" in cls:
                self._in_poem_line = True
                self._current_line = ""
            elif "p3" in cls:
                # Stanza break
                self.lines.append("")
        elif tag == "span" and self._in_poem_line:
            pass  # content comes in handle_data

    def handle_endtag(self, tag):
        if tag == "div" and self._in_title:
            self._in_title = False
        elif tag == "div" and self._in_header:
            self._in_header = False
        elif tag == "p" and self._in_poem_line:
            self._in_poem_line = False
            self.lines.append(self._current_line.strip())
            self._current_line = ""

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._in_header:
            self.date += data
        elif self._in_poem_line:
            self._current_line += data


def extract_poem(html_path: Path) -> dict:
    """Extract a single poem from an HTML file."""
    parser = PoemExtractor()
    parser.feed(html_path.read_text(encoding="utf-8"))

    # Strip zero-width and invisible unicode chars from line ends
    lines = [re.sub(r"[\u200b-\u200f\u2028-\u202f\ufeff]+", "", ln) for ln in parser.lines]

    # Trim trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    # Trim leading empty lines
    while lines and not lines[0].strip():
        lines.pop(0)

    text = "\n".join(lines)

    return {
        "text": text,
        "title": parser.title.strip(),
        "author": "Quinn",
        "date": parser.date.strip(),
        "source": "quinn_original",
    }


def main():
    mypoems_dir = Path("data/mypoems")
    output_file = Path("data/processed/quinn_poems.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    html_files = sorted(mypoems_dir.glob("*.html"))
    if not html_files:
        print(f"No HTML files found in {mypoems_dir}")
        return

    poems = []
    for path in html_files:
        poem = extract_poem(path)
        if poem["text"].strip():
            poems.append(poem)
            print(f"  {poem['title']} ({len(poem['text'].split(chr(10)))} lines)")
        else:
            print(f"  SKIP (empty): {path.name}")

    with open(output_file, "w") as f:
        for poem in poems:
            f.write(json.dumps(poem) + "\n")

    print(f"\nExtracted {len(poems)} poems to {output_file}")


if __name__ == "__main__":
    main()
