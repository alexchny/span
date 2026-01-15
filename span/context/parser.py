import ast
from pathlib import Path


def extract_imports_ast(file_path: Path) -> list[str]:
    try:
        content = file_path.read_text()
        tree = ast.parse(content, filename=str(file_path))
    except Exception:
        return []

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports


def compute_file_hash(file_path: Path) -> str:
    import hashlib

    content = file_path.read_text()
    return hashlib.sha256(content.encode()).hexdigest()
