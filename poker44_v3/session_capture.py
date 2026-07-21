"""Capture v3.0 session payloads and distil the telemetry vocabulary.

This is the highest-value thing the staged miner does on flip day.  The
concrete telemetry ``event_type`` / ``target`` / ``value`` vocabulary is
documented NOWHERE in the subnet repository -- the only example in the entire
tree is the string ``"pointer_click"`` in one test fixture.  The first
tournament window (Weekly Poker Championship, 2026-07-24 18:00 UTC) is the
first and only place to observe it.  Whoever reverse-engineers that vocabulary
first owns the format reset.

Two artefacts per request:
  * ``sessions_<window>_<ts>.json``  -- the full raw payload (rotated), which
    is what we will train on once labels can be inferred/joined.
  * ``vocab_<window>_<ts>.json``     -- a small distilled vocabulary report,
    cheap to keep forever and cheap to diff between windows.

Never raises: diagnostics must not be able to break serving.
"""

import json
import os
import time

from poker44_v3 import session_features as sf

DEFAULT_DIR = "/root/bittensor/poker44-data/v3_capture"
DEFAULT_MAX_PAYLOADS = 8


class SessionCapture:
    def __init__(self, directory=None, max_payloads=None, logger=None):
        self.dir = directory or os.getenv("POKER44_V3_CAPTURE_DIR", DEFAULT_DIR)
        try:
            self.max_payloads = int(
                max_payloads if max_payloads is not None
                else os.getenv("POKER44_V3_CAPTURE_MAX", DEFAULT_MAX_PAYLOADS))
        except (TypeError, ValueError):
            self.max_payloads = DEFAULT_MAX_PAYLOADS
        self.enabled = os.getenv("POKER44_V3_CAPTURE", "1") != "0"
        self.log = logger or (lambda level, msg: None)
        self.windows_seen = set()

    def _rotate(self, pattern):
        try:
            files = sorted(
                f for f in os.listdir(self.dir) if f.startswith(pattern))
            while len(files) >= self.max_payloads:
                os.unlink(os.path.join(self.dir, files.pop(0)))
        except Exception:  # noqa: BLE001
            pass

    def record(self, window_id, sessions, scores, info):
        if not self.enabled:
            return
        try:
            os.makedirs(self.dir, exist_ok=True)
            stamp = int(time.time())
            safe_window = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in str(window_id))[:64]

            vocab = sf.telemetry_vocabulary(sessions)
            vocab["window_id"] = window_id
            vocab["captured_at"] = stamp
            vocab["n_sessions"] = len(sessions)
            vocab["session_top_level_keys"] = sorted(
                {k for s in sessions if isinstance(s, dict) for k in s})
            vocab["hand_keys"] = sorted({
                k for s in sessions if isinstance(s, dict)
                for h in (s.get("hands") or []) if isinstance(h, dict) for k in h})
            vocab["action_keys"] = sorted({
                k for s in sessions if isinstance(s, dict)
                for h in (s.get("hands") or []) if isinstance(h, dict)
                for a in (h.get("actions") or []) if isinstance(a, dict) for k in a})
            vocab["telemetry_keys"] = sorted({
                k for s in sessions if isinstance(s, dict)
                for k in (s.get("telemetry") or {})
                if isinstance(s.get("telemetry"), dict)})
            vocab["summary_keys"] = sorted({
                k for s in sessions if isinstance(s, dict)
                if isinstance(s.get("telemetry"), dict)
                for k in (s["telemetry"].get("summary") or {})})
            vocab["score_summary"] = {
                "path": info.get("path"),
                "active_signals": info.get("active_signals"),
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "mean": (sum(scores) / len(scores)) if scores else None,
            }
            with open(os.path.join(
                    self.dir, "vocab_%s_%d.json" % (safe_window, stamp)), "w") as fh:
                json.dump(vocab, fh, default=str)

            self._rotate("sessions_")
            with open(os.path.join(
                    self.dir, "sessions_%s_%d.json" % (safe_window, stamp)), "w") as fh:
                json.dump({"window_id": window_id, "captured_at": stamp,
                           "scores": scores, "sessions": sessions}, fh, default=str)

            if window_id not in self.windows_seen:
                self.windows_seen.add(window_id)
                self.log("info",
                         "V3 CAPTURE: first sight of window=%s | %d sessions | "
                         "telemetry event_types=%s"
                         % (window_id, len(sessions),
                            [t for t, _ in vocab["telemetry_event_types"][:12]]))
        except Exception as exc:  # noqa: BLE001
            self.log("warning", "v3 capture failed: %r" % (exc,))
