#!/usr/bin/env python3
"""Validate local links and anchors in a built Sphinx HTML tree."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, _tag, attrs):
        values = dict(attrs)
        if values.get("href") is not None:
            self.hrefs.append(values["href"])
        if values.get("id"):
            self.anchors.add(values["id"])
        if values.get("name"):
            self.anchors.add(values["name"])


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def main() -> int:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("html_root", type=Path)
    args = argparser.parse_args()

    html_root = args.html_root.resolve()
    if not html_root.is_dir():
        raise SystemExit(f"Built HTML directory does not exist: {html_root}")

    pages = {path.resolve(): parse_page(path) for path in html_root.rglob("*.html")}
    failures = []
    checked = 0
    for source, parsed in pages.items():
        for href in parsed.hrefs:
            url = urlsplit(href)
            if url.scheme or url.netloc or href.startswith("//"):
                continue
            path_part = unquote(url.path)
            if path_part.startswith("/"):
                target = html_root / path_part.lstrip("/")
            elif path_part:
                target = source.parent / path_part
            else:
                target = source
            target = target.resolve()
            if target.is_dir():
                target = target / "index.html"
            checked += 1
            try:
                target.relative_to(html_root)
            except ValueError:
                failures.append(
                    f"{source.relative_to(html_root)} -> {href} leaves HTML root"
                )
                continue
            if not target.exists():
                failures.append(f"{source.relative_to(html_root)} -> {href} is missing")
                continue
            if url.fragment and target.suffix == ".html":
                target_parser = pages.get(target)
                if target_parser is None:
                    target_parser = parse_page(target)
                    pages[target] = target_parser
                fragment = unquote(url.fragment)
                if fragment not in target_parser.anchors:
                    failures.append(
                        f"{source.relative_to(html_root)} -> {href} "
                        "has no target anchor"
                    )

    if failures:
        print("Local HTML link validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        "Local HTML link validation passed for "
        f"{checked} links in {len(pages)} pages."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
