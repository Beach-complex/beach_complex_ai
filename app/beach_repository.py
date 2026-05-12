import json
from pathlib import Path
from typing import Dict, List, Optional

from app.schemas import BeachInfo


class BeachRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._beaches = self._load(path)

    def list_all(self) -> List[BeachInfo]:
        return list(self._beaches.values())

    def get(self, beach_id: str) -> Optional[BeachInfo]:
        return self._beaches.get(self.normalize_code(beach_id))

    def codes(self) -> List[str]:
        return sorted(self._beaches.keys())

    @staticmethod
    def normalize_code(beach_id: str) -> str:
        return beach_id.upper()

    @staticmethod
    def _load(path: Path) -> Dict[str, BeachInfo]:
        with path.open(encoding="utf-8") as fp:
            raw = json.load(fp)

        beaches: Dict[str, BeachInfo] = {}
        for entry in raw["beaches"]:
            code = entry["code"].upper()
            beaches[code] = BeachInfo(
                id=code,
                name=entry["name"],
                status=entry["status"],
                lat=float(entry["lat"]),
                lon=float(entry["lon"]),
                popularity_weight=float(entry["popularity_weight"]),
            )
        return beaches
