"""lerobot-record with line-based episode control (works in any terminal, no raw key capture).

While an episode records: Enter = episode done, "r" + Enter = redo it, "q" + Enter = stop.
Takes the same arguments as lerobot-record; called by tools/record_real.sh.
"""

import sys
import threading

import lerobot.scripts.lerobot_record as rec
from lerobot.utils.keyboard_input import apply_recording_control


class _LineListener:
    def stop(self):
        pass


def _init_line_listener():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def read_lines():
        for line in sys.stdin:
            key = line.strip().lower()
            if key in ("", "n"):
                apply_recording_control("right", events)
            elif key == "r":
                apply_recording_control("left", events)
            elif key == "q":
                apply_recording_control("esc", events)

    threading.Thread(target=read_lines, daemon=True).start()
    print(">>> Enter = Episode fertig | r + Enter = neu aufnehmen | q + Enter = stopp", flush=True)
    return _LineListener(), events


rec.init_keyboard_listener = _init_line_listener
sys.exit(rec.main())
