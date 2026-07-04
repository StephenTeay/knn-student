"""
db/db_init.py
=============
Creates (or resets) the SQLite database from schema.sql.

Run directly:  python -m db.db_init [--reset]

DECISION: foreign_keys enforcement is OFF by default in SQLite for
backward-compatibility reasons baked into the engine itself; we turn it on
explicitly on every connection (here and in every other module that opens
a connection) so that referential integrity actually behaves the way the
schema declares it should -- otherwise a bug elsewhere could silently
insert a Surveys row for a research_code that doesn't exist in Students.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, SCHEMA_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Single place that opens a SQLite connection with FKs enforced.
    Every other module imports this instead of calling sqlite3.connect
    directly, so the PRAGMA is never accidentally forgotten."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(reset: bool = False) -> None:
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[db_init] removed existing database at {DB_PATH}")

    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    conn.close()
    print(f"[db_init] database ready at {DB_PATH}")
    print(f"[db_init] tables: {[t[0] for t in tables]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop and recreate the database file")
    args = parser.parse_args()
    init_db(reset=args.reset)
