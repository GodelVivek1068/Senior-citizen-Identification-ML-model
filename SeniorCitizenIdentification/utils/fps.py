"""
utils/fps.py
============================================================
A simple rolling-average FPS counter used to display real-time
performance metrics in the UI and on-screen overlay.
============================================================
"""

import time
from collections import deque


class FPSCounter:
    def __init__(self, window_size: int = 30):
        """
        Parameters
        ----------
        window_size : int
            Number of most recent frame timestamps used to
            compute the rolling average FPS.
        """
        self.window_size = window_size
        self._timestamps = deque(maxlen=window_size)

    def tick(self) -> float:
        """
        Call once per processed frame.

        Returns
        -------
        float : current rolling-average FPS.
        """
        now = time.time()
        self._timestamps.append(now)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed
