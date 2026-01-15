import sqlite3
import time
from pathlib import Path

from span.context.parser import compute_file_hash, extract_imports_ast
from span.context.repo_map import RepoMap


def test_extract_imports_ast(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
import os
import sys
from pathlib import Path
from typing import Any, List
from mymodule import something
""")

    imports = extract_imports_ast(test_file)

    assert "os" in imports
    assert "sys" in imports
    assert "pathlib" in imports
    assert "typing" in imports
    assert "mymodule" in imports


def test_extract_imports_invalid_file(tmp_path: Path) -> None:
    test_file = tmp_path / "invalid.py"
    test_file.write_text("this is not valid python <<<")

    imports = extract_imports_ast(test_file)

    assert imports == []


def test_compute_file_hash(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    hash1 = compute_file_hash(test_file)
    hash2 = compute_file_hash(test_file)

    assert hash1 == hash2
    assert len(hash1) == 64

    test_file.write_text("print('world')")
    hash3 = compute_file_hash(test_file)

    assert hash3 != hash1


def test_repo_map_init(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)

    assert db_path.exists()
    repo_map.close()


def test_repo_map_update_file(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)

    timestamp = int(time.time())
    repo_map.update_file(
        file_path="src/main.py",
        file_hash="abc123",
        imports=["os", "sys", "pathlib"],
        timestamp=timestamp,
    )

    file_hash = repo_map.get_file_hash("src/main.py")
    assert file_hash == "abc123"

    repo_map.close()


def test_repo_map_update_file_overwrites(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)

    timestamp = int(time.time())
    repo_map.update_file(
        file_path="src/main.py",
        file_hash="abc123",
        imports=["os"],
        timestamp=timestamp,
    )

    repo_map.update_file(
        file_path="src/main.py",
        file_hash="def456",
        imports=["sys"],
        timestamp=timestamp + 1,
    )

    file_hash = repo_map.get_file_hash("src/main.py")
    assert file_hash == "def456"

    repo_map.close()


def test_repo_map_resolve_dependencies(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)

    timestamp = int(time.time())

    repo_map.update_file(
        file_path="mymodule/__init__.py",
        file_hash="hash1",
        imports=[],
        timestamp=timestamp,
    )

    repo_map.update_file(
        file_path="src/main.py",
        file_hash="hash2",
        imports=["mymodule", "os"],
        timestamp=timestamp,
    )

    repo_map.resolve_dependencies(tmp_path)

    cursor = repo_map.conn.cursor()
    cursor.execute("SELECT source_file, target_file FROM dependencies")
    deps = cursor.fetchall()

    assert ("src/main.py", "mymodule/__init__.py") in deps

    repo_map.close()


def test_repo_map_find_affected_tests(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)

    timestamp = int(time.time())

    repo_map.update_file(
        file_path="src/auth.py",
        file_hash="hash1",
        imports=[],
        timestamp=timestamp,
    )

    repo_map.update_file(
        file_path="tests/test_auth.py",
        file_hash="hash2",
        imports=["src.auth"],
        timestamp=timestamp,
    )

    repo_map.update_file(
        file_path="tests/test_other.py",
        file_hash="hash3",
        imports=["src.other"],
        timestamp=timestamp,
    )

    repo_map.resolve_dependencies(tmp_path)

    affected = repo_map.find_affected_tests(
        modified_files=["src/auth.py"],
        test_patterns=["tests/"],
    )

    assert "tests/test_auth.py" in affected
    assert "tests/test_other.py" not in affected

    repo_map.close()


def test_repo_map_find_affected_tests_includes_modified_tests(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)

    timestamp = int(time.time())

    repo_map.update_file(
        file_path="tests/test_feature.py",
        file_hash="hash1",
        imports=[],
        timestamp=timestamp,
    )

    affected = repo_map.find_affected_tests(
        modified_files=["tests/test_feature.py"],
        test_patterns=["tests/"],
    )

    assert "tests/test_feature.py" in affected

    repo_map.close()


def test_repo_map_context_manager(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"

    with RepoMap(db_path) as repo_map:
        timestamp = int(time.time())
        repo_map.update_file(
            file_path="test.py",
            file_hash="hash",
            imports=[],
            timestamp=timestamp,
        )

    assert db_path.exists()


def test_foreign_key_enforcement(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)
    cursor = repo_map.conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO imports (source_file, imported_module) VALUES (?, ?)",
            ("nonexistent.py", "os")
        )
        repo_map.conn.commit()
        raise AssertionError("Foreign key constraint should have prevented this insert")
    except sqlite3.IntegrityError:
        pass
    finally:
        repo_map.close()


def test_find_affected_tests_empty_patterns(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)
    timestamp = int(time.time())

    repo_map.update_file("src/auth.py", "hash1", [], timestamp)
    repo_map.update_file("tests/test_auth.py", "hash2", ["src.auth"], timestamp)
    repo_map.resolve_dependencies(tmp_path)

    affected = repo_map.find_affected_tests(["src/auth.py"], [])

    assert "tests/test_auth.py" in affected
    repo_map.close()


def test_find_affected_tests_directory_pattern(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)
    timestamp = int(time.time())

    repo_map.update_file("src/module.py", "hash1", [], timestamp)
    repo_map.update_file("tests/test_module.py", "hash2", ["src.module"], timestamp)
    repo_map.update_file("my_tests/test_other.py", "hash3", ["src.module"], timestamp)
    repo_map.resolve_dependencies(tmp_path)

    affected = repo_map.find_affected_tests(["src/module.py"], ["tests/"])

    assert "tests/test_module.py" in affected
    assert "my_tests/test_other.py" not in affected
    repo_map.close()


def test_find_affected_tests_filename_pattern(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    repo_map = RepoMap(db_path)
    timestamp = int(time.time())

    repo_map.update_file("src/module.py", "hash1", [], timestamp)
    repo_map.update_file("test_module.py", "hash2", ["src.module"], timestamp)
    repo_map.update_file("tests/my_test_file.py", "hash3", ["src.module"], timestamp)
    repo_map.resolve_dependencies(tmp_path)

    affected = repo_map.find_affected_tests(["src/module.py"], ["test_"])

    assert "test_module.py" in affected
    assert "tests/my_test_file.py" not in affected
    repo_map.close()
