"""Offline source normalization; unsupported expansion is reported, never hidden."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import re
import textwrap
import unicodedata
from html.parser import HTMLParser


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output = []
        self.in_article = False
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skip += 1
        if self.in_article and not self.skip:
            if tag in {"p", "li", "h1", "h2", "h3", "h4", "pre", "tr", "dt", "dd", "br"}:
                self.output.append("\n")
            if tag in {"td", "th"}:
                self.output.append(" | ")

    def handle_endtag(self, tag):
        if self.in_article and not self.skip and tag in {"p", "li", "pre", "tr", "dt", "dd"}:
            self.output.append("\n")
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skip = max(0, self.skip - 1)
        if tag == "article":
            self.in_article = False

    def handle_data(self, text):
        if self.in_article and not self.skip:
            self.output.append(text)


def normalize(text):
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip() + "\n"


def python_docs(text, module):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    result = []

    def visit(node, prefix):
        for child in getattr(node, "body", []):
            if not isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or child.name.startswith("_"):
                continue
            symbol = prefix + "." + child.name
            doc = ast.get_docstring(child, clean=True)
            if doc:
                signature = "(" + ast.unparse(child.args) + ")" if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else ""
                result.append({"symbol": symbol, "text": symbol + signature + "\n\n" + doc,
                               "line": child.lineno, "end_line": child.end_lineno})
            if isinstance(child, ast.ClassDef):
                visit(child, symbol)
    visit(tree, module)
    return result


class Resolver:
    def __init__(self, root, api, substitutions=None):
        self.root = Path(root).resolve()
        self.api = api
        self.gaps = []
        self.origins = set()
        self.substitutions = substitutions or {}

    def gap(self, source, kind, target):
        item = {"path": source, "kind": kind, "target": target}
        if item not in self.gaps:
            self.gaps.append(item)

    def read(self, relative, stack=()):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or relative in stack:
            self.gap(stack[-1] if stack else relative, "include", relative)
            return "[UNRESOLVED INCLUDE: " + relative + "]"
        self.origins.add(relative)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            self.gap(relative, "encoding", "not UTF-8")
            return "[UNRESOLVED ENCODING: " + relative + "]"
        if path.suffix == ".html":
            parser = HTMLText(); parser.feed(text)
            result = "".join(parser.output)
            if not result.strip():
                self.gap(relative, "html_extraction", "article_body_not_found")
                return text
            return normalize(result)
        if path.suffix == ".ipynb":
            try:
                book = json.loads(text)
                return "\n\n".join("".join(cell.get("source", [])) if cell.get("cell_type") != "code" else "```python\n" + "".join(cell.get("source", [])) + "\n```" for cell in book.get("cells", []))
            except (ValueError, TypeError):
                self.gap(relative, "notebook", "invalid notebook")
                return text
        def include(match):
            kind, target, options = match.group(1), match.group(2).strip(), match.group(3) or ""
            if "{{" in target or "://" in target:
                self.gap(relative, kind, target)
                return match.group(0)
            if target.startswith("/"):
                child = target.lstrip("/")
            else:
                child = str((path.parent / target).resolve().relative_to(self.root)) if (path.parent / target).resolve().is_relative_to(self.root) else "../outside"
            content = self.read(child, (*stack, relative))
            options_map = dict(re.findall(r"^\s+:([\w-]+):\s*(.*?)\s*$", options, re.M))
            try:
                if "lines" in options_map:
                    lines, selected = content.splitlines(keepends=True), []
                    for spec in options_map["lines"].split(","):
                        a, sep, b = spec.strip().partition("-")
                        lo, hi = int(a or 1), int(b or len(lines)) if sep else int(a)
                        selected.extend(lines[lo-1:hi])
                    content = "".join(selected)
                for option, keep in (("start-after", False), ("start-at", True)):
                    if option in options_map:
                        marker = options_map[option]; pos = content.index(marker)
                        content = content[pos if keep else pos + len(marker):]
                for option, keep in (("end-before", False), ("end-at", True)):
                    if option in options_map:
                        marker = options_map[option]; pos = content.index(marker)
                        content = content[:pos + (len(marker) if keep else 0)]
                unsupported = set(options_map) - {"lines", "start-after", "start-at", "end-before", "end-at", "language", "linenos", "dedent", "caption", "name", "class", "encoding", "emphasize-lines", "number-lines"}
                for opt in sorted(unsupported):
                    self.gap(relative, kind + "_option", opt + "=" + options_map[opt])
                if "dedent" in options_map:
                    content = textwrap.dedent(content)
            except (ValueError, IndexError):
                self.gap(relative, kind + "_selector", options)
            return ("```\n" + content + "\n```") if kind == "literalinclude" else content
        text = re.sub(r"^\s*\.\. (include|literalinclude)::\s*([^\n]+)\n((?:[ \t]+[^\n]*\n|\n)*)", include, text + "\n", flags=re.M)
        def md_include(match):
            target = match.group(1).split("#")[0]
            child = (path.parent / target).resolve()
            if not child.is_relative_to(self.root):
                self.gap(relative, "include", target)
                return match.group(0)
            return self.read(str(child.relative_to(self.root)), (*stack, relative))
        text = re.sub(r"\{%\s*include\s*\[[^]]*\]\(([^)]+)\)\s*%\}", md_include, text)
        module = ""
        roots = {symbol.split(".")[0] for symbol in self.api}
        def qualify(name, current):
            name = name.lstrip("~")
            if name in self.api or name.split(".")[0] in roots:
                return name
            return current + "." + name if current else name
        def autodoc(match):
            nonlocal module
            kind, name = match.group(1), match.group(2).strip()
            if kind in {"currentmodule", "module"}:
                module = name
                return match.group(0)
            symbol = qualify(name, module)
            prefix = ""
            if kind == "automodule":
                module = symbol
                prefix = "\n.. currentmodule:: " + symbol + "\n\n"
            record = self.api.get(symbol)
            if record:
                return prefix + record["text"]
            if kind == "automodule":
                entries = [v["text"] for k, v in sorted(self.api.items()) if k.startswith(symbol + ".") and "." not in k[len(symbol)+1:]]
                if entries:
                    return prefix + "\n\n".join(entries)
            self.gap(relative, kind, symbol)
            return prefix + match.group(0)
        text = re.sub(r"^\s*\.\. (?:py:)?(currentmodule|module|autoclass|autofunction|automethod|autoattribute|automodule|autodata|autoproperty)::\s*([^\n]+)", autodoc, text, flags=re.M)
        # Autosummary entries name public API, including compiled objects. Keep
        # the list and append exact installed descriptions if available.
        additions = []
        for match in re.finditer(r"\.\. autosummary::\s*\n((?:[ \t]+[^\n]*\n|\n)+)", text):
            preceding = re.findall(r"\.\. (?:py:)?(?:currentmodule|module)::\s*([^\n]+)", text[:match.start()])
            current = preceding[-1].strip() if preceding else ""
            for line in match.group(1).splitlines():
                symbol = line.strip().lstrip("~")
                if not symbol or symbol.startswith(":") or " " in symbol:
                    continue
                qualified = qualify(symbol, current)
                if qualified in self.api:
                    additions.append(self.api[qualified]["text"])
                else:
                    self.gap(relative, "autosummary", qualified)
        text += "\n\n" + "\n\n".join(dict.fromkeys(additions))
        # Generated substitutions and executable directive outputs cannot be
        # called resolved just because their source text exists.
        for _ in range(10):
            expanded = re.sub(r"\{\{\s*([^}\s]+)\s*\}\}", lambda match: str(self.substitutions.get(match.group(1), match.group(0))), text)
            if expanded == text:
                break
            text = expanded
        for expr in sorted(set(re.findall(r"\{\{[^}]+\}\}", text))):
            self.gap(relative, "substitution", expr)
        for kind in re.findall(r"^\s*\.\. (?:[\w-]+:)?(eval-rst|exec|jupyter-execute|plot|ipython)::", text, re.M):
            self.gap(relative, "generated_output", kind)
        return normalize(text)


def required_spans(text):
    """Known warning blocks are addressable and cannot become implicit permission."""
    patterns = [r"(?m)^\s*\.\. (?:warning|caution|danger|important|note)::[^\n]*(?:\n(?:[ \t]+[^\n]*|[ \t]*))*",
                r"(?ms)^Warnings?\n[-=]{3,}\n.*?(?=\n[A-Z][^\n]*\n[-=]{3,}|\Z)",
                r"(?s)\{%\s*note\s+(?:warning|alert|info)[^%]*%\}.*?\{%\s*endnote\s*%\}"]
    ranges = sorted({(match.start(), match.end()) for pattern in patterns for match in re.finditer(pattern, text) if match.end() > match.start()})
    merged = []
    for start, end in ranges:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start,end))
    return [{"start": start, "end": end, "kind": "warning_with_condition"} for start,end in merged]
