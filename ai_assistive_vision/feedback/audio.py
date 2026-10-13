"""Phase 7 audio feedback module."""

import logging
import queue
import threading
from typing import Optional

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


LOGGER = logging.getLogger(__name__)


class AudioEngine:
    """Non-blocking audio engine for text-to-speech feedback."""

    def __init__(self, rate: int = 180, enabled: bool = True):
        self.enabled = enabled and TTS_AVAILABLE
        self.rate = rate
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        if not TTS_AVAILABLE and enabled:
            LOGGER.warning("pyttsx3 is not installed. Audio feedback is disabled.")

    def start(self) -> None:
        """Start the background TTS thread."""
        if not self.enabled:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background TTS thread."""
        self._running = False
        # Push empty string to wake up worker
        if self.enabled:
            self._queue.put("")
        if self._thread:
            self._thread.join(timeout=1.0)

    def speak(self, text: str) -> None:
        """Queue a string to be spoken."""
        if not self.enabled or not text:
            return
        
        # Keep queue small to reduce latency/stale speech
        if self._queue.qsize() < 2:
            self._queue.put(text)

    def generate_event_text(self, event: dict) -> str:
        """Convert a deduplicated event dictionary into a short phrase."""
        event_name = str(event.get("event", "obstacle"))
        label = str(event.get("object", "obstacle")).capitalize()
        direction = str(event.get("direction", "ahead"))

        if direction == "center":
            direction_phrase = "ahead"
        else:
            direction_phrase = f"on the {direction}"

        if event_name == "text":
            text_val = str(event.get("text", "text")).title()
            return f"{text_val} {direction_phrase}."
        if event_name == "approaching":
            return f"{label} approaching {direction_phrase}."
        if event_name == "receding":
            return f"{label} moving away {direction_phrase}."
        if event_name == "disappearing":
            return f"{label} cleared."
        if event_name == "direction_change":
            return f"{label} moving {direction_phrase}."
        if event_name == "door":
            return f"Door {direction_phrase}."

        return f"{label} {direction_phrase}."



    def _worker(self) -> None:
        """Background worker that initializes pyttsx3 and consumes the queue."""
        try:
            # Initialize engine in the thread that uses it
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            
            while self._running:
                try:
                    text = self._queue.get(timeout=0.1)
                    if text and self._running:
                        engine.say(text)
                        engine.runAndWait()
                except queue.Empty:
                    continue
        except Exception as e:
            LOGGER.error(f"TTS Engine error: {e}")
            self._running = False
