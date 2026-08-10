"""로컬 parquet 캐시 공용 유틸 — data_loader, screener 가 공유."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from . import config


def cache_path(kind: str, key: str) -> Path:
    """키를 해시해 캐시 파일 경로를 만든다 (심볼에 '/' 등이 섞여도 안전)."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return config.CACHE_DIR / f"{kind}_{digest}.parquet"


def is_fresh(path: Path, ttl_sec: int = config.CACHE_TTL_SEC) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_sec


def clear_cache() -> int:
    """캐시 파일을 모두 지우고 삭제한 개수를 반환한다."""
    files = list(config.CACHE_DIR.glob("*.parquet"))
    for f in files:
        f.unlink()
    return len(files)
