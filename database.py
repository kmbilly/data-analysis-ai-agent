from typing import Any, Dict, List, Optional
import sqlite3

# ---------------------------
# SQLite helpers (read‑only)
# ---------------------------
class Database:
    def __init__(self, path: str):
        # Note: SQLite has no strict read‑only mode without opening URI flags;
        # we still enforce query_only and guard against writes.
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only = ON;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def schema_overview(self, max_cols: int = 200) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT name, type FROM sqlite_master WHERE type in ('table','view') ORDER BY name;")
        items = cur.fetchall()
        lines: List[str] = []
        for row in items:
            name = row["name"]
            typ = row["type"]
            cols = self.columns(name)
            col_str = ", ".join([f"{c['name']}:{c['type']}" for c in cols])
            if len(col_str) > max_cols:
                col_str = col_str[:max_cols] + " …"
            lines.append(f"- {typ} {name} (columns: {col_str})")
        if not lines:
            return "(No tables found)"
        return "\n".join(lines)

    def columns(self, table: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        return [
            {"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "default": r[4], "pk": r[5]}
            for r in cur.fetchall()
        ]

    def run_sql(self, sql: str, params: Optional[tuple] = None, row_limit: int = 1000) -> Dict[str, Any]:
        # Guard against writes
        lowered = sql.strip().lower()
        forbidden = ("insert", "update", "delete", "drop", "alter", "create", "replace", "attach", "detach", "vacuum", "pragma")
        if lowered.split()[0] in forbidden:
            raise ValueError("Write or unsafe statement is not allowed.")
        cur = self.conn.cursor()
        cur.execute(sql, params or tuple())
        rows = cur.fetchmany(row_limit)
        cols = [d[0] for d in cur.description] if cur.description else []
        data = [dict(zip(cols, r)) for r in rows]
        return {"columns": cols, "rows": data, "row_count": len(data)}
