"""
capture.py — Webcam Capture Module
===================================
Provides live webcam capture with real-time FPS display.
Exposes both a standalone preview mode and a generator
for downstream consumers (landmark detection, inference).
"""

import cv2
import time


class WebcamCapture:
    """Manages webcam video capture with FPS tracking."""

    def __init__(self, camera_index=0, window_name="Sign Language Translator"):
        """
        Initialise the webcam capture.

        Args:
            camera_index: Index of the camera device (default 0).
            window_name: Title for the OpenCV display window.
        """
        self.camera_index = camera_index
        self.window_name = window_name
        self.cap = None
        self._prev_time = 0
        self._fps = 0.0

    def start(self):
        """Open the webcam connection."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. "
                "Check that your webcam is connected and not in use by another application."
            )
        # Set preferred resolution (camera will use closest supported)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._prev_time = time.perf_counter()
        return self

    def stop(self):
        """Release the webcam and close all windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def _update_fps(self):
        """Calculate the current frames per second."""
        current_time = time.perf_counter()
        elapsed = current_time - self._prev_time
        if elapsed > 0:
            self._fps = 1.0 / elapsed
        self._prev_time = current_time

    def draw_fps(self, frame):
        """
        Overlay the FPS counter on the frame.

        Args:
            frame: BGR image (numpy array) to draw on.

        Returns:
            The frame with FPS text overlaid.
        """
        fps_text = f"FPS: {self._fps:.1f}"
        # Black background rectangle for readability
        cv2.rectangle(frame, (10, 10), (170, 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            fps_text,
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),  # Green text
            2,
            cv2.LINE_AA,
        )
        return frame

    @property
    def fps(self):
        """Current frames per second."""
        return self._fps

    def get_frames(self):
        """
        Generator that yields frames from the webcam.

        Yields:
            Tuple of (success: bool, frame: numpy array).
            Frame is a BGR image. Stops when camera fails.
        """
        if self.cap is None:
            self.start()

        while True:
            success, frame = self.cap.read()
            if not success:
                print("[WARNING] Failed to read frame from webcam.")
                break

            # Flip horizontally for mirror-view (more natural for sign language)
            frame = cv2.flip(frame, 1)
            self._update_fps()
            yield success, frame

    def preview(self):
        """
        Run a standalone webcam preview with FPS display.
        Press 'q' to quit.
        """
        print(f"[INFO] Starting webcam preview (camera index: {self.camera_index})")
        print("[INFO] Press 'q' to quit.")

        self.start()

        try:
            for success, frame in self.get_frames():
                self.draw_fps(frame)
                cv2.imshow(self.window_name, frame)

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Quit signal received.")
                    break
        finally:
            self.stop()


# ─── Standalone Usage ───────────────────────────────────────────────
if __name__ == "__main__":
    cam = WebcamCapture(camera_index=0)
    cam.preview()
