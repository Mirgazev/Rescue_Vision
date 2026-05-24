import csv
import time
from pathlib import Path
from typing import Dict, Iterable


class MetricsLogger:
    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def write(self, row: Dict):
        data = {name: row.get(name, "") for name in self.fieldnames}
        data.setdefault("timestamp", time.time())
        self._writer.writerow(data)
        self._file.flush()

    def close(self):
        self._file.close()
