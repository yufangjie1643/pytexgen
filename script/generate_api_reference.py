"""Generate a compact API reference from Python docstrings and C++ comments.

The SWIG generated ``Core.py`` has weak Python-level docstrings. This script
reads the SWIG include list in ``Python/Core.i`` and the C++ Doxygen comments in
``Core/*.h`` so the public Python package can ship a readable API reference
without editing generated files.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "api_reference.md"


@dataclass
class PythonApi:
    module: str
    kind: str
    name: str
    signature: str
    doc: str


@dataclass
class CppApi:
    header: str
    kind: str
    name: str
    declaration: str
    doc: str


def _clean_doc(text: str) -> str:
    """Return a readable Markdown-ready docstring/comment block."""
    text = inspect.cleandoc(text)
    if not text:
        return ""
    return text


def _humanize(name: str) -> str:
    """Convert a CamelCase API name into readable words."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip().lower() or name


def _fallback_doc(name: str, owner: str = "this API") -> str:
    """Create a readable fallback when the source has no Doxygen comment."""
    verbs = {
        "Get": "Return {subject} from {owner}.",
        "Set": "Set {subject} on {owner}.",
        "Add": "Add {subject} to {owner}.",
        "Delete": "Delete {subject} from {owner}.",
        "Remove": "Remove {subject} from {owner}.",
        "Clear": "Clear {subject} on {owner}.",
        "Save": "Save {subject} for {owner}.",
        "Read": "Read {subject} into {owner}.",
        "Create": "Create {subject} for {owner}.",
        "Build": "Build {subject} for {owner}.",
        "Assign": "Assign {subject} to {owner}.",
        "Output": "Output {subject} from {owner}.",
        "Rotate": "Rotate {owner}.",
        "Translate": "Translate {owner}.",
        "Copy": "Return a copy of {owner}.",
        "Populate": "Populate {subject} from {owner}.",
        "Insert": "Insert {subject} into {owner}.",
        "Replace": "Replace {subject} on {owner}.",
        "Swap": "Swap {subject} on {owner}.",
        "Move": "Move {subject} on {owner}.",
        "Convert": "Convert {subject} for {owner}.",
        "Detect": "Detect {subject} for {owner}.",
        "Valid": "Return whether {owner} is valid.",
    }
    for verb, template in verbs.items():
        if name.startswith(verb):
            return template.format(subject=_humanize(name[len(verb):]), owner=owner)
    return f"SWIG-wrapped TexGen C++ API entry for `{name}`."


def _python_signature(node: ast.AST) -> str:
    """Build a readable Python function/class signature from an AST node."""
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(positional, defaults):
        text = arg.arg
        if default is not None:
            text += "=..."
        args.append(text)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = arg.arg
        if default is not None:
            text += "=..."
        args.append(text)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"def {node.name}({', '.join(args)})"


def iter_python_api(paths: list[Path]) -> list[PythonApi]:
    """Collect top-level and class-level Python APIs from source files."""
    entries: list[PythonApi] = []
    for path in paths:
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                entries.append(
                    PythonApi(
                        module=module,
                        kind="function",
                        name=node.name,
                        signature=_python_signature(node),
                        doc=_clean_doc(ast.get_docstring(node) or ""),
                    )
                )
            elif isinstance(node, ast.ClassDef):
                entries.append(
                    PythonApi(
                        module=module,
                        kind="class",
                        name=node.name,
                        signature=_python_signature(node),
                        doc=_clean_doc(ast.get_docstring(node) or ""),
                    )
                )
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        entries.append(
                            PythonApi(
                                module=module,
                                kind="method",
                                name=f"{node.name}.{child.name}",
                                signature=_python_signature(child),
                                doc=_clean_doc(ast.get_docstring(child) or ""),
                            )
                        )
    return entries


def swig_core_headers() -> list[Path]:
    """Return C++ headers included by ``Python/Core.i``."""
    core_i = ROOT / "Python" / "Core.i"
    headers = []
    for rel in re.findall(r'%include\s+"\.\./Core/([^"]+\.h)"', core_i.read_text(encoding="utf-8")):
        path = ROOT / "Core" / rel
        if path.exists():
            headers.append(path)
    return headers


def _clean_comment_line(line: str) -> str:
    """Strip Doxygen markers from one C++ comment line."""
    line = line.strip()
    line = re.sub(r"^///\s?", "", line)
    line = re.sub(r"^/\*\*\s?", "", line)
    line = re.sub(r"^\*/\s?", "", line)
    line = re.sub(r"^\*\s?", "", line)
    line = line.replace("\\param", "param").replace("\\return", "return")
    return line.strip()


def _compact_cpp_declaration(lines: list[str]) -> str:
    """Normalize a possibly multiline C++ declaration to one line."""
    return " ".join(" ".join(lines).replace("\t", " ").split())


def _cpp_name_from_decl(declaration: str) -> tuple[str, str] | None:
    """Extract kind and display name from a C++ class/function declaration."""
    class_match = re.search(r"\b(class|struct)\s+(?:CLASS_DECLSPEC\s+)?([A-Za-z_]\w*)", declaration)
    if class_match:
        if "{" not in declaration:
            return None
        return "class", class_match.group(2)
    if "(" not in declaration or ")" not in declaration:
        return None
    if any(declaration.startswith(prefix) for prefix in ("if ", "for ", "while ", "switch ")):
        return None
    before = declaration.split("(", 1)[0].strip()
    if not before:
        return None
    name = before.split()[-1].replace("*", "").replace("&", "")
    if name in {"operator"}:
        return None
    return "function", name


def iter_cpp_api(headers: list[Path]) -> list[CppApi]:
    """Collect classes and declarations from C++ headers."""
    entries: list[CppApi] = []
    for header in headers:
        rel_header = str(header.relative_to(ROOT)).replace("\\", "/")
        comment: list[str] = []
        in_block_comment = False
        pending_decl: list[str] = []
        current_class = "this class"

        for raw_line in header.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue

            if in_block_comment:
                comment.append(_clean_comment_line(stripped))
                if "*/" in stripped:
                    in_block_comment = False
                continue
            if stripped.startswith("/**"):
                in_block_comment = "*/" not in stripped
                comment.append(_clean_comment_line(stripped))
                continue
            if stripped.startswith("///"):
                comment.append(_clean_comment_line(stripped))
                continue
            if stripped.startswith("//"):
                continue

            code = stripped.split("//", 1)[0].strip()
            if not code:
                continue

            starts_decl = (
                "CLASS_DECLSPEC" in code
                or code.startswith(("class ", "struct ", "virtual ", "static ", "const "))
                or bool(re.match(r"^[A-Za-z_][\w:<>,~*&\s]+\s+[A-Za-z_~]\w*\s*\(", code))
            )
            if pending_decl or starts_decl:
                pending_decl.append(code)
                if ";" in code or "{" in code:
                    declaration = _compact_cpp_declaration(pending_decl)
                    pending_decl = []
                    parsed = _cpp_name_from_decl(declaration)
                    if parsed:
                        kind, name = parsed
                        doc = _clean_doc("\n".join(comment))
                        if kind == "class":
                            current_class = name
                        if not doc:
                            doc = _fallback_doc(name, current_class)
                        entries.append(
                            CppApi(
                                header=rel_header,
                                kind=kind,
                                name=name,
                                declaration=declaration,
                                doc=doc,
                            )
                        )
                    comment = []
            else:
                comment = []
    return entries


def write_markdown(path: Path, python_entries: list[PythonApi], cpp_entries: list[CppApi]) -> None:
    """Write the API reference Markdown document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PyTexGen API Reference",
        "",
        "This file is generated by `script/generate_api_reference.py` from Python docstrings and C++ header comments.",
        "Regenerate it after changing public APIs or header comments.",
        "",
        "## Python Modules",
        "",
    ]

    current_module = None
    for entry in python_entries:
        if entry.module != current_module:
            current_module = entry.module
            lines.extend([f"### `{current_module}`", ""])
        lines.extend(
            [
                f"#### `{entry.name}`",
                "",
                f"```python\n{entry.signature}\n```",
                "",
                entry.doc,
                "",
            ]
        )

    lines.extend(["## SWIG Core C++ API", ""])
    current_header = None
    for entry in cpp_entries:
        if entry.header != current_header:
            current_header = entry.header
            lines.extend([f"### `{current_header}`", ""])
        lang = "cpp"
        lines.extend(
            [
                f"#### `{entry.name}`",
                "",
                f"```{lang}\n{entry.declaration}\n```",
                "",
                entry.doc,
                "",
            ]
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    """Generate ``docs/api_reference.md``."""
    python_paths = [
        ROOT / "TexGen" / "gpu_voxelizer.py",
        ROOT / "src" / "pytexgen" / "__init__.py",
        ROOT / "src" / "pytexgen" / "_core_docs.py",
    ]
    python_entries = iter_python_api([path for path in python_paths if path.exists()])
    cpp_entries = iter_cpp_api(swig_core_headers())
    write_markdown(DEFAULT_OUTPUT, python_entries, cpp_entries)
    print(f"Wrote {DEFAULT_OUTPUT.relative_to(ROOT)}")
    print(f"Python entries: {len(python_entries)}")
    print(f"C++ entries: {len(cpp_entries)}")


if __name__ == "__main__":
    main()
