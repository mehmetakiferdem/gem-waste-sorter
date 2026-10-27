"""Turns per-frame (present, label, confidence) into one decision per item.

States:
    EMPTY    tray empty, waiting for an item
    SETTLE   item just appeared, the hand is probably still in the picture
    VOTE     collecting `window` frames of classifier output
    WAIT     decided, waiting for the tray to be emptied again
"""

from collections import Counter
from dataclasses import dataclass

EMPTY, SETTLE, VOTE, WAIT = "BOS", "YERLESIYOR", "OYLAMA", "BEKLE"
UNSURE = "emin_degil"


@dataclass
class Decision:
    label: str          # bin key, or UNSURE
    agreement: float    # winner's share of the counted votes
    confidence: float   # winner's mean confidence


class DecisionMaker:
    def __init__(self, settle_frames=8, window=12, min_confidence=0.55,
                 min_agreement=0.6, empty_frames=10, stuck_frames=90):
        self.settle_frames = settle_frames
        self.window = window
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.empty_frames = empty_frames
        self.stuck_frames = stuck_frames
        # Set when the scene never emptied after a decision (camera moved,
        # light changed, item left in place): the caller should relearn the
        # empty scene. Cleared by the caller.
        self.needs_rebase = False
        self.state = EMPTY
        self._stuck = 0
        self._count = 0
        self._stuck = 0
        self._votes = []

    def update(self, present, label=None, confidence=0.0):
        """Feed one frame. Returns a Decision on the frame one is made."""
        if self.state == EMPTY:
            if present:
                self._go(SETTLE if self.settle_frames else VOTE)
            return None

        if self.state == SETTLE:
            if not present:
                self._go(EMPTY)
            else:
                self._count += 1
                if self._count >= self.settle_frames:
                    self._go(VOTE)
            return None

        if self.state == VOTE:
            if not present:
                # Taken away before we decided.
                self._go(EMPTY)
                return None
            self._count += 1
            if label is not None and confidence >= self.min_confidence:
                self._votes.append((label, confidence))
            if self._count < self.window:
                return None
            decision = self._tally()
            self._go(WAIT)
            return decision

        # WAIT: the tray must stay empty for a while before the next item.
        if present:
            self._count = 0
            self._stuck += 1
            if self.stuck_frames and self._stuck >= self.stuck_frames:
                self.needs_rebase = True
                self._go(EMPTY)
        else:
            self._count += 1
            if self._count >= self.empty_frames:
                self._go(EMPTY)
        return None

    def _tally(self):
        if not self._votes:
            return Decision(UNSURE, 0.0, 0.0)
        winner, n = Counter(l for l, _ in self._votes).most_common(1)[0]
        agreement = n / len(self._votes)
        conf = sum(c for l, c in self._votes if l == winner) / n
        if agreement < self.min_agreement:
            return Decision(UNSURE, agreement, conf)
        return Decision(winner, agreement, conf)

    def _go(self, state):
        self.state = state
        self._count = 0
        self._stuck = 0
        self._votes = []
