"""Load markdown guidelines and split them into searchable sections."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Section:
    id: str       # for example "sentiment#sarcasm-and-irony"
    title: str
    text: str


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def split_sections(stem: str, markdown: str) -> list[Section]:
    """Every '## Heading' starts a section. Text before the first heading is ignored."""
    sections: list[Section] = []
    title = None
    lines: list[str] = []

    def flush():
        if title is not None:
            body = "\n".join(lines).strip()
            if body:
                sections.append(Section(id=f"{stem}#{slugify(title)}", title=title, text=body))

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            title = line[3:].strip()
            lines = []
        elif title is not None:
            lines.append(line)
    flush()
    return sections


def load_guidelines(folder: str | Path) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(Path(folder).glob("*.md")):
        sections.extend(split_sections(path.stem, path.read_text(encoding="utf-8")))
    return sections
