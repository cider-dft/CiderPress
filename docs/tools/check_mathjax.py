#!/usr/bin/env python3
"""Validate rendered math and reject obvious unprocessed math source."""

from __future__ import annotations

import argparse
import functools
from html.parser import HTMLParser
import http.server
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from urllib.parse import quote


MATH_MARKER = 'class="math notranslate nohighlight"'
ERROR_MARKERS = ("data-mjx-error=", "<mjx-merror")
RAW_MATH_PATTERNS = (
    re.compile(
        r"\b[A-Za-z][A-Za-z0-9]*(?:\([^)]*\))?\s*\^\s*"
        r"(?:\{?[-+]?\d+|[A-Za-z])"
    ),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*\s*\*\*\s*[-+A-Za-z0-9(]"),
    re.compile(
        r"\\(?:frac|int|sum|mathbf|mathrm|text|nabla|partial|exp|sqrt|"
        r"alpha|beta|gamma|sigma|tau|rho)\b"
    ),
    re.compile(r"\$[^$\n]+\$"),
)


class RenderedTextParser(HTMLParser):
    """Collect visible math-like text outside code and math elements."""

    def __init__(self):
        super().__init__()
        self.stack: list[tuple[str, bool]] = []
        self.failures: list[str] = []

    def handle_starttag(self, tag, attrs):
        classes = set(dict(attrs).get("class", "").split())
        suppressed = tag in {"code", "pre", "script", "style"} or "math" in classes
        self.stack.append((tag, suppressed))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if any(suppressed for _tag, suppressed in self.stack):
            return
        text = " ".join(data.split())
        if text and any(pattern.search(text) for pattern in RAW_MATH_PATTERNS):
            self.failures.append(text)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format, *args):
        pass


def find_browser() -> str:
    configured = os.environ.get("MATHJAX_BROWSER")
    candidates = [configured] if configured else []
    candidates.extend(["google-chrome", "chromium", "chromium-browser"])
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise SystemExit(
        "MathJax check requires google-chrome, chromium, or chromium-browser "
        "(or set MATHJAX_BROWSER)."
    )


def pages_with_math(html_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(html_root.rglob("*.html"))
        if MATH_MARKER in path.read_text(encoding="utf-8")
    ]


def find_unrendered_math(html_root: Path) -> list[str]:
    failures = []
    for path in sorted(html_root.rglob("*.html")):
        parser = RenderedTextParser()
        parser.feed(path.read_text(encoding="utf-8"))
        relative = path.relative_to(html_root)
        failures.extend(f"{relative}: {text}" for text in parser.failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html_root", type=Path)
    args = parser.parse_args()

    html_root = args.html_root.resolve()
    if not html_root.is_dir():
        raise SystemExit(f"Built HTML directory does not exist: {html_root}")

    pages = pages_with_math(html_root)
    if not pages:
        raise SystemExit(f"No rendered-math pages found under {html_root}")

    raw_math_failures = find_unrendered_math(html_root)
    if raw_math_failures:
        print("Unrendered mathematical notation found outside code elements:")
        for failure in raw_math_failures:
            print(f"  - {failure}")
        return 1

    browser = find_browser()
    handler = functools.partial(QuietHandler, directory=str(html_root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    failures = []
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        for page in pages:
            relative = page.relative_to(html_root).as_posix()
            url = f"{base_url}/{quote(relative)}"
            result = subprocess.run(
                [
                    browser,
                    "--headless",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=5000",
                    "--dump-dom",
                    url,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            rendered = result.stdout
            reasons = []
            if result.returncode:
                reasons.append(f"browser exited with status {result.returncode}")
            if "<mjx-container" not in rendered:
                reasons.append("MathJax did not produce rendered output")
            if any(marker in rendered for marker in ERROR_MARKERS):
                reasons.append("MathJax emitted an error element")
            if reasons:
                failures.append(f"{relative}: {', '.join(reasons)}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    if failures:
        print("Rendered MathJax validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"Rendered MathJax validation passed for {len(pages)} pages using "
        f"{browser}; no obvious unrendered math was found."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
