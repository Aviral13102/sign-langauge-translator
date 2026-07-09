"""
tts.py -- Text-to-Speech Module
=================================
Background-threaded TTS using pyttsx3 for offline voice readout
of translated words.

Usage:
    from src.tts import TTSEngine

    tts = TTSEngine()
    tts.speak("hello")
    tts.shutdown()
"""

import threading
import queue


class TTSEngine:
    """
    Non-blocking text-to-speech engine using pyttsx3.

    Runs pyttsx3 in a dedicated background thread to avoid blocking
    the main webcam/inference loop. Queues speech requests and
    processes them sequentially.

    Args:
        rate: Speech rate in words per minute (default: 150).
        volume: Volume from 0.0 to 1.0 (default: 0.9).
        enabled: Whether TTS is active (default: True).
    """

    def __init__(self, rate=150, volume=0.9, enabled=True):
        self._rate = rate
        self._volume = volume
        self._enabled = enabled
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self._engine = None

        if enabled:
            self.start()

    def start(self):
        """Start the background TTS thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background worker thread that processes the speech queue."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
        except Exception as e:
            print(f"[WARNING] TTS engine failed to initialise: {e}")
            print("[INFO] TTS will be disabled.")
            self._enabled = False
            self._running = False
            return

        print("[INFO] TTS engine started (pyttsx3)")

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    break  # Shutdown signal

                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"[WARNING] TTS speak error: {e}")

                self._queue.task_done()

            except queue.Empty:
                continue

        # Cleanup
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass

    def speak(self, text):
        """
        Queue text for speech output.

        Args:
            text: The text string to speak.
        """
        if not self._enabled or not self._running:
            return

        if not text or not text.strip():
            return

        self._queue.put(text.strip())

    def toggle(self):
        """
        Toggle TTS on/off.

        Returns:
            True if TTS is now enabled, False if disabled.
        """
        if self._enabled:
            self._enabled = False
            # Clear any pending speech
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            return False
        else:
            self._enabled = True
            if not self._running:
                self.start()
            return True

    @property
    def enabled(self):
        """Whether TTS is currently enabled."""
        return self._enabled

    def shutdown(self):
        """Stop the TTS engine and background thread."""
        self._running = False
        self._queue.put(None)  # Shutdown signal

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._thread = None
        self._engine = None

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass


# ---- Quick Test ------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  [ASL] TTS Engine -- Quick Test")
    print("=" * 60)

    tts = TTSEngine(rate=150, volume=0.9)
    time.sleep(1)  # Let engine initialise

    print("[TEST] Speaking 'hello world'...")
    tts.speak("hello world")

    print("[TEST] Speaking 'A S L translator ready'...")
    tts.speak("A S L translator ready")

    # Wait for queue to drain
    time.sleep(5)

    print("[TEST] Toggling TTS off...")
    state = tts.toggle()
    print(f"  TTS enabled: {state}")

    print("[TEST] This should NOT be spoken...")
    tts.speak("this should be silent")

    print("[TEST] Toggling TTS back on...")
    state = tts.toggle()
    print(f"  TTS enabled: {state}")

    print("[TEST] Speaking 'goodbye'...")
    tts.speak("goodbye")
    time.sleep(3)

    tts.shutdown()
    print("[INFO] TTS test complete.")
    print("=" * 60)
