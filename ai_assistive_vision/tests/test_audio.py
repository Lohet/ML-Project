"""Tests for Phase 7 audio feedback."""

from unittest.mock import MagicMock, patch

from ai_assistive_vision.feedback.audio import AudioEngine


def test_audio_engine_generate_event_text():
    engine = AudioEngine(enabled=False)
    
    event = {"object": "person", "direction": "center"}
    assert engine.generate_event_text(event) == "Person ahead."
    
    event2 = {"object": "car", "direction": "left"}
    assert engine.generate_event_text(event2) == "Car on the left."


def test_audio_engine_does_not_block():
    engine = AudioEngine(enabled=False)
    
    # Should not block or raise error when disabled
    engine.start()
    engine.speak("Test")
    engine.stop()


@patch("ai_assistive_vision.feedback.audio.TTS_AVAILABLE", True)
@patch("ai_assistive_vision.feedback.audio.pyttsx3", create=True)
def test_audio_engine_worker(mock_pyttsx3):
    mock_engine = MagicMock()
    mock_pyttsx3.init.return_value = mock_engine
    
    engine = AudioEngine(enabled=True)
    engine.start()
    
    engine.speak("Test speech")
    
    # Wait briefly for worker to pick up task
    import time
    time.sleep(0.2)
    engine.stop()
    
    # Verify that the mocked TTS engine was called
    mock_engine.say.assert_called_with("Test speech")
    mock_engine.runAndWait.assert_called()
