import sqlite3
from pathlib import Path
from typing import Any


class RepoMap:
    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path.cwd() / ".span" / "repo.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_indexed INTEGER NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imports (
                source_file TEXT NOT NULL,
                imported_module TEXT NOT NULL,
                FOREIGN KEY (source_file) REFERENCES files(path)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dependencies (
                source_file TEXT NOT NULL,
                target_file TEXT NOT NULL,
                FOREIGN KEY (source_file) REFERENCES files(path),
                FOREIGN KEY (target_file) REFERENCES files(path)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_imports_source ON imports(source_file)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(imported_module)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_deps_target ON dependencies(target_file)
        """)

        self.conn.commit()

    def update_file(
        self,
        file_path: str,
        file_hash: str,
        imports: list[str],
        timestamp: int,
    ) -> None:
        cursor = self.conn.cursor()

        cursor.execute(
            "DELETE FROM imports WHERE source_file = ?",
            (file_path,)
        )
        cursor.execute(
            "DELETE FROM dependencies WHERE source_file = ?",
            (file_path,)
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO files (path, hash, last_indexed)
            VALUES (?, ?, ?)
            """,
            (file_path, file_hash, timestamp),
        )

        for imported_module in imports:
            cursor.execute(
                "INSERT INTO imports (source_file, imported_module) VALUES (?, ?)",
                (file_path, imported_module),
            )

        self.conn.commit()

    def resolve_dependencies(self, project_root: Path) -> None:
        cursor = self.conn.cursor()

        cursor.execute("DELETE FROM dependencies")

        cursor.execute("SELECT source_file, imported_module FROM imports")
        imports_data = cursor.fetchall()

        cursor.execute("SELECT path FROM files")
        all_files = {row[0] for row in cursor.fetchall()}

        for source_file, imported_module in imports_data:
            module_parts = imported_module.split(".")
            possible_paths = [
                f"{'/'.join(module_parts)}.py",
                f"{'/'.join(module_parts)}/__init__.py",
            ]

            for possible_path in possible_paths:
                if possible_path in all_files:
                    cursor.execute(
                        "INSERT INTO dependencies (source_file, target_file) VALUES (?, ?)",
                        (source_file, possible_path),
                    )
                    break

        self.conn.commit()

    def find_affected_tests(
        self,
        modified_files: list[str],
        test_patterns: list[str],
    ) -> list[str]:
        if not modified_files:
            return []

        cursor = self.conn.cursor()
        affected_tests = set()

        query = f"""
            SELECT DISTINCT source_file
            FROM dependencies
            WHERE target_file IN ({','.join('?' * len(modified_files))})
        """

        cursor.execute(query, modified_files)

        for row in cursor.fetchall():
            source_file = row[0]
            if not test_patterns or self._matches_test_pattern(source_file, test_patterns):
                affected_tests.add(source_file)

        for modified_file in modified_files:
            if self._matches_test_pattern(modified_file, test_patterns):
                affected_tests.add(modified_file)

        return sorted(affected_tests)

    def _matches_test_pattern(self, file_path: str, test_patterns: list[str]) -> bool:
        if not test_patterns:
            return False

        for pattern in test_patterns:
            if pattern.endswith("/"):
                if file_path.startswith(pattern):
                    return True
                if f"/{pattern}" in file_path:
                    return True
            else:
                basename = file_path.split("/")[-1]
                if basename.startswith(pattern):
                    return True
                if file_path.startswith(pattern):
                    return True

        return False

    def get_file_hash(self, file_path: str) -> str | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT hash FROM files WHERE path = ?", (file_path,))
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RepoMap":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
