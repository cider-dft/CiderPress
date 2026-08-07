import os
from pathlib import Path
from urllib.parse import quote

from sphinx.util.nodes import split_explicit_title
from docutils import nodes, utils

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_REF = os.environ.get("CIDERPRESS_SOURCE_REF") or os.environ.get(
    "GITHUB_SHA", "main"
)


def _source_uri(target):
    ref = quote(SOURCE_REF, safe="")
    path = quote(target, safe="/")
    return f"https://github.com/mir-group/CiderPress/blob/{ref}/{path}"


def source_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    has_title, title, target = split_explicit_title(text)
    title = utils.unescape(title)
    target = utils.unescape(target)
    source_path = (REPOSITORY_ROOT / target).resolve()
    try:
        source_path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        message = inliner.reporter.error(
            f"Source link escapes the repository: {target}", line=lineno
        )
        return [inliner.problematic(rawtext, rawtext, message)], [message]
    if not source_path.is_file():
        message = inliner.reporter.error(
            f"Source link target does not exist: {target}", line=lineno
        )
        return [inliner.problematic(rawtext, rawtext, message)], [message]
    refnode = nodes.reference(title, title, refuri=_source_uri(target))
    return [refnode], []

def setup(app):
    app.add_role('source', source_role)
    return {'version': '0.1', 'parallel_read_safe': True}
