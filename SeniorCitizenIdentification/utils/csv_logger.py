"""
utils/csv_logger.py
============================================================
Handles all visit-logging to CSV (and optional Excel export).

Deduplication rules implemented (per project spec)
---------------------------------------------------
1. A given tracking ID is logged AT MOST ONCE while it remains
   continuously visible (no duplicate rows per frame).
2. If a tracking ID is not seen for longer than
   config.REAPPEAR_TIMEOUT_SECONDS, it is considered "gone".
   If that same ID (or, more commonly, a new ID assigned to the
   same physical person after the tracker's own buffer expired)
   appears again afterwards, it is logged again as a new visit.
3. Only senior citizens (age > config.AGE_THRESHOLD) trigger a
   log entry, per the "log only when a new senior citizen first
   appears" requirement. Set `log_all_visitors=True` to log
   everyone instead.
============================================================
"""

import csv
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class _TrackState:
    last_seen: datetime
    logged: bool = False
    first_logged_at: Optional[datetime] = None


class CSVLogger:
    def __init__(
        self,
        csv_path: str = config.CSV_PATH,
        reappear_timeout_seconds: int = config.REAPPEAR_TIMEOUT_SECONDS,
        log_all_visitors: bool = False,
    ):
        self.csv_path = csv_path
        self.reappear_timeout_seconds = reappear_timeout_seconds
        self.log_all_visitors = log_all_visitors

        self._lock = threading.Lock()
        self._track_states: Dict[int, _TrackState] = {}
        self._serial_counter = 0

        self._ensure_csv_exists()
        self._serial_counter = self._count_existing_rows()

    # ------------------------------------------------------------------
    def _ensure_csv_exists(self):
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.isfile(self.csv_path):
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(config.CSV_COLUMNS)
        except OSError as exc:
            raise RuntimeError(
                f"[CSVLogger] Could not create/access CSV file at "
                f"'{self.csv_path}': {exc}"
            ) from exc

    def _count_existing_rows(self) -> int:
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)  # minus header
        except OSError:
            return 0

    # ------------------------------------------------------------------
    def update_seen(self, track_id: int):
        """Call every frame for every currently-tracked ID (senior or not)
        so we can correctly detect reappearance-after-timeout."""
        now = datetime.now()
        with self._lock:
            state = self._track_states.get(track_id)
            if state is None:
                self._track_states[track_id] = _TrackState(last_seen=now)
                return

            # If this ID had gone quiet longer than the timeout, treat
            # its return as a fresh visit (reset the 'logged' flag).
            gap = (now - state.last_seen).total_seconds()
            if gap > self.reappear_timeout_seconds:
                state.logged = False

            state.last_seen = now

    def log_if_new(
        self,
        track_id: int,
        age: int,
        gender: str,
        is_senior: bool,
    ) -> bool:
        """
        Logs a visit row if this track ID has not already been logged
        for its current "visit session".

        Returns
        -------
        bool : True if a new row was actually written.
        """
        if not self.log_all_visitors and not is_senior:
            return False

        with self._lock:
            state = self._track_states.get(track_id)
            if state is None:
                state = _TrackState(last_seen=datetime.now())
                self._track_states[track_id] = state

            if state.logged:
                return False  # already logged for this visit session

            now = datetime.now()
            self._serial_counter += 1

            row = [
                self._serial_counter,
                f"ID-{track_id}",
                age,
                gender,
                "Yes" if is_senior else "No",
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                now.strftime("%Y-%m-%d %H:%M:%S"),
            ]

            try:
                with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except OSError as exc:
                logger.error(f"[CSVLogger] Failed to write row: {exc}")
                self._serial_counter -= 1  # roll back counter on failure
                return False

            state.logged = True
            state.first_logged_at = now
            logger.info(f"Logged visit: {row}")
            return True

    # ------------------------------------------------------------------
    def read_log_dataframe(self) -> pd.DataFrame:
        """Returns the full log as a pandas DataFrame (used by the UI)."""
        try:
            return pd.read_csv(self.csv_path)
        except (OSError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=config.CSV_COLUMNS)

    def export_to_excel(self, excel_path: Optional[str] = None) -> str:
        """Exports the current CSV log to an .xlsx file. Returns the path."""
        if excel_path is None:
            excel_path = self.csv_path.replace(".csv", ".xlsx")
        df = self.read_log_dataframe()
        try:
            df.to_excel(excel_path, index=False, engine="openpyxl")
        except OSError as exc:
            raise RuntimeError(f"[CSVLogger] Failed to export Excel file: {exc}") from exc
        return excel_path
