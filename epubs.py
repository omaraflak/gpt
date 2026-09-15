import io
import os
import re
import html
import zipfile
import xml.etree.ElementTree as ET

_HEAD_RE = re.compile(r"<head\b[^>]*>.*?</head>", re.S | re.I)
_STYLE_SCRIPT_RE = re.compile(r"<(style|script)\b[^>]*>.*?</\1>", re.S | re.I)
_BOILER_RE = re.compile(
    r'<div\b[^>]*class="[^"]*pg-boilerplate[^"]*"[^>]*>.*', re.S | re.I
)
_BLOCK_TAG_RE = re.compile(
    r"</?(p|div|h[1-6]|br|li|tr|hr|blockquote|section|article)\b[^>]*>", re.I
)
_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NL_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def _html_to_text(html_str: str) -> str:
    """Convert an HTML/XHTML chapter string from an EPUB into clean plain text."""
    html_str = _HEAD_RE.sub("", html_str)
    html_str = _STYLE_SCRIPT_RE.sub("", html_str)
    html_str = _BOILER_RE.sub("", html_str)
    html_str = _BLOCK_TAG_RE.sub("\n", html_str)
    text = _TAG_RE.sub("", html_str)
    text = html.unescape(text)

    for marker in (
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ):
        if marker in text:
            text = text.split(marker)[0]

    for marker in (
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ):
        if marker in text:
            parts = text.split(marker, 1)
            nl = parts[1].find("\n")
            text = parts[1][nl + 1 :] if nl != -1 else parts[1]

    text = _MULTI_NL_RE.sub("\n\n", text).strip()
    paras = text.split("\n\n")
    while paras and any(
        k in paras[0]
        for k in (
            "Produced by",
            "Distributed Proofreaders",
            "Online Distributed",
            "Project Gutenberg",
        )
    ):
        paras.pop(0)
    return "\n\n".join(paras)


def _extract_text_from_zip(z: zipfile.ZipFile) -> str:
    """Extract ordered plain text from an open EPUB ZipFile."""
    container = ET.fromstring(z.read("META-INF/container.xml"))
    rootfile = next(el for el in container.iter() if el.tag.endswith("rootfile"))
    opf_path = rootfile.attrib["full-path"]
    opf_dir = os.path.dirname(opf_path)
    opf = ET.fromstring(z.read(opf_path))

    manifest = {
        el.attrib["id"]: el.attrib["href"]
        for el in opf.iter()
        if el.tag.endswith("item") and "id" in el.attrib and "href" in el.attrib
    }
    spine = [
        el.attrib["idref"]
        for el in opf.iter()
        if el.tag.endswith("itemref") and "idref" in el.attrib
    ]

    book_parts = []
    namelist = set(z.namelist())
    for idref in spine:
        if idref in ("coverpage-wrapper", "pg-footer"):
            continue
        href = manifest.get(idref)
        if not href:
            continue
        full_href = f"{opf_dir}/{href}" if opf_dir else href
        if full_href in namelist:
            raw_html = z.read(full_href).decode("utf-8", errors="ignore")
            cleaned = _html_to_text(raw_html)
            if cleaned:
                book_parts.append(cleaned)

    return "\n\n".join(book_parts)


def extract_text(epub: str | os.PathLike | bytes) -> str:
    """Extract plain text from an .epub file path or raw bytes."""
    source = io.BytesIO(epub) if isinstance(epub, bytes) else epub
    with zipfile.ZipFile(source, "r") as z:
        return _extract_text_from_zip(z)
