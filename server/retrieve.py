"""Hybrid retrieval over poe2.db: semantic (sqlite-vec) + keyword (FTS5), fused
with Reciprocal Rank Fusion. PoE item/skill/keystone names are exact-match-y, so
the FTS half matters as much as the vector half.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass

import sqlite_vec

RRF_K = 60
POOL = 40


@dataclass
class Hit:
    id: int
    source: str
    title: str
    url: str
    section: str
    text: str
    score: float


def connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def _fts_query(q: str) -> str:
    # keep alphanumerics, OR the terms so partial matches still rank
    terms = re.findall(r"[A-Za-z0-9]+", q)
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


def search(db: sqlite3.Connection, query_vec: list[float], query_text: str, k: int = 8) -> list[Hit]:
    vblob = struct.pack(f"{len(query_vec)}f", *query_vec)
    vec_ids = [
        r[0]
        for r in db.execute(
            "SELECT rowid FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vblob, POOL),
        )
    ]
    try:
        fts_ids = [
            r[0]
            for r in db.execute(
                "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
                (_fts_query(query_text), POOL),
            )
        ]
    except sqlite3.OperationalError:
        fts_ids = []

    ranks: dict[int, float] = {}
    for lst in (vec_ids, fts_ids):
        for rank, cid in enumerate(lst):
            ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    top = sorted(ranks, key=ranks.get, reverse=True)[:k]
    if not top:
        return []
    rows = {
        r["id"]: r
        for r in db.execute(
            f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(top))})", top
        )
    }
    return [
        Hit(
            id=cid,
            source=rows[cid]["source"],
            title=rows[cid]["title"],
            url=rows[cid]["url"],
            section=rows[cid]["section"],
            text=rows[cid]["text"],
            score=ranks[cid],
        )
        for cid in top
        if cid in rows
    ]
