"""
word_builder.py -- Letter-to-Word Assembly Pipeline
=====================================================
Manages prediction smoothing (majority-vote buffer), hold-to-confirm
logic, word assembly, and sentence/history tracking.

The pipeline is:
    raw frame predictions → majority vote → hold-to-confirm → word buffer → history

Usage:
    from src.word_builder import WordBuilder

    wb = WordBuilder()
    wb.add_prediction("A", 0.95)
    state = wb.get_state()
"""

import time
from collections import Counter, deque


class WordBuilder:
    """
    Assembles letters into words from noisy per-frame predictions.

    Args:
        buffer_size: Number of frames for majority-vote smoothing (default: 12).
        confirm_seconds: Seconds a letter must be stable to be confirmed (default: 0.5).
        confidence_threshold: Minimum confidence to accept a prediction (default: 0.85).
        fps: Expected frames per second for hold-to-confirm calculation (default: 30).
    """

    def __init__(
        self,
        buffer_size=12,
        confirm_seconds=0.5,
        confidence_threshold=0.85,
        fps=30,
    ):
        self.buffer_size = buffer_size
        self.confirm_seconds = confirm_seconds
        self.confidence_threshold = confidence_threshold
        self.fps = fps

        # Majority-vote buffer
        self._vote_buffer = deque(maxlen=buffer_size)

        # Hold-to-confirm state
        self._current_candidate = None
        self._candidate_start_time = None
        self._last_confirmed = None

        # Word assembly
        self._current_word = []
        self._history = []  # List of completed words

        # Statistics
        self._total_predictions = 0
        self._confirmed_letters = 0

    # ---- Public API --------------------------------------------------------

    def add_prediction(self, letter, confidence):
        """
        Feed a single frame's prediction into the pipeline.

        Args:
            letter: Predicted class label (e.g. "A", "space", "del").
            confidence: Prediction confidence (0.0 to 1.0).

        Returns:
            The confirmed letter if one was just confirmed, else None.
        """
        self._total_predictions += 1

        # Filter low-confidence predictions
        if confidence < self.confidence_threshold:
            self._vote_buffer.append(None)
            return None

        self._vote_buffer.append(letter)

        # Majority vote over the buffer
        majority = self._get_majority()
        if majority is None:
            self._reset_candidate()
            return None

        # Hold-to-confirm logic
        now = time.time()

        if majority != self._current_candidate:
            # New candidate letter
            self._current_candidate = majority
            self._candidate_start_time = now
            return None

        # Same candidate — check if hold time exceeded
        elapsed = now - self._candidate_start_time
        if elapsed >= self.confirm_seconds:
            # Prevent re-confirming the same letter immediately
            if majority == self._last_confirmed:
                return None

            # Letter confirmed!
            self._last_confirmed = majority
            self._confirmed_letters += 1
            self._handle_confirmed(majority)
            # Reset candidate timer so next hold starts fresh
            self._candidate_start_time = now
            return majority

        return None

    def add_no_hand(self):
        """
        Call when no hand is detected in the frame.
        Fills the buffer with None to decay the majority vote.
        """
        self._vote_buffer.append(None)
        self._reset_candidate()

    def confirm_word(self):
        """
        Confirm the current word and move it to history.
        Called when user presses SPACE or the 'space' gesture is detected.

        Returns:
            The completed word string, or None if current word is empty.
        """
        if not self._current_word:
            return None

        word = "".join(self._current_word)
        self._history.append(word)
        self._current_word = []
        self._last_confirmed = None
        return word

    def delete_last(self):
        """Delete the last letter from the current word (backspace)."""
        if self._current_word:
            self._current_word.pop()
            self._last_confirmed = None

    def clear_word(self):
        """Clear the current word without adding to history."""
        self._current_word = []
        self._last_confirmed = None

    def clear_history(self):
        """Clear all history and current word."""
        self._current_word = []
        self._history = []
        self._last_confirmed = None
        self._reset_candidate()
        self._vote_buffer.clear()

    def get_state(self):
        """
        Get the full state of the word builder for HUD rendering.

        Returns:
            Dictionary with current state.
        """
        return {
            "current_letter": self._current_candidate,
            "current_word": "".join(self._current_word),
            "history": list(self._history),
            "full_text": self._get_full_text(),
            "hold_progress": self._get_hold_progress(),
            "buffer_majority": self._get_majority(),
            "is_confirming": self._current_candidate is not None,
            "stats": {
                "total_predictions": self._total_predictions,
                "confirmed_letters": self._confirmed_letters,
                "words_completed": len(self._history),
            },
        }

    def get_current_word(self):
        """Get the current word being assembled."""
        return "".join(self._current_word)

    def get_full_text(self):
        """Get full text including history and current word."""
        return self._get_full_text()

    def get_hold_progress(self):
        """Get hold-to-confirm progress (0.0 to 1.0)."""
        return self._get_hold_progress()

    # ---- Internal Methods --------------------------------------------------

    def _get_majority(self):
        """
        Compute majority vote from the buffer.

        Returns:
            The most common non-None prediction, or None if no clear majority.
        """
        valid = [p for p in self._vote_buffer if p is not None]
        if not valid:
            return None

        counter = Counter(valid)
        most_common, count = counter.most_common(1)[0]

        # Require at least 50% of buffer to agree
        if count >= len(self._vote_buffer) * 0.5:
            return most_common

        return None

    def _get_hold_progress(self):
        """
        Calculate hold-to-confirm progress as a fraction.

        Returns:
            Float between 0.0 and 1.0.
        """
        if self._current_candidate is None or self._candidate_start_time is None:
            return 0.0

        elapsed = time.time() - self._candidate_start_time
        progress = min(elapsed / self.confirm_seconds, 1.0)
        return progress

    def _handle_confirmed(self, letter):
        """Process a confirmed letter — add to word or handle special classes."""
        if letter == "space":
            self.confirm_word()
        elif letter == "del":
            self.delete_last()
        else:
            self._current_word.append(letter)

    def _reset_candidate(self):
        """Reset the hold-to-confirm candidate."""
        self._current_candidate = None
        self._candidate_start_time = None

    def _get_full_text(self):
        """Build full text from history + current word."""
        parts = list(self._history)
        current = "".join(self._current_word)
        if current:
            parts.append(current)
        return " ".join(parts)


# ---- Quick Test ------------------------------------------------------------

if __name__ == "__main__":
    import time as _time

    wb = WordBuilder(buffer_size=5, confirm_seconds=0.3)

    print("=" * 60)
    print("  [ASL] WordBuilder -- Quick Test")
    print("=" * 60)

    # Simulate a sequence of predictions
    test_sequence = [
        # 10 frames of "H" at high confidence
        ("H", 0.95), ("H", 0.93), ("H", 0.96), ("H", 0.94), ("H", 0.95),
        ("H", 0.92), ("H", 0.93), ("H", 0.95), ("H", 0.94), ("H", 0.96),
    ]

    print("\n  Simulating 10 frames of 'H' with 0.3s confirm time...")
    for i, (letter, conf) in enumerate(test_sequence):
        result = wb.add_prediction(letter, conf)
        _time.sleep(0.05)  # ~20fps
        if result:
            print(f"  Frame {i+1}: CONFIRMED -> '{result}'")
        else:
            state = wb.get_state()
            print(f"  Frame {i+1}: candidate='{state['current_letter']}', "
                  f"progress={state['hold_progress']:.0%}")

    state = wb.get_state()
    print(f"\n  Final state:")
    print(f"    Current word: '{state['current_word']}'")
    print(f"    History: {state['history']}")
    print(f"    Full text: '{state['full_text']}'")
    print(f"    Stats: {state['stats']}")
    print("=" * 60)
