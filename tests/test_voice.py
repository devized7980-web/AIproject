import time
import threading
import types
import unittest

import importlib.util
import os

# Load main.py as a module via importlib to avoid import path issues in tests
spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
main = importlib.util.module_from_spec(spec)
assert spec is not None
import sys
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


class MockEngine:
    def __init__(self, recorder: list, raise_on: set | None = None):
        self.recorder = recorder
        self.raise_on = raise_on or set()
        self.stopped = False

    def setProperty(self, *_):
        pass

    def say(self, text):
        if text in self.raise_on:
            raise RuntimeError("TTS failure")
        self.recorder.append(text)

    def runAndWait(self):
        time.sleep(0.01)

    def stop(self):
        self.stopped = True


class EngineModule:
    def __init__(self, recorder: list, raise_on: set | None = None):
        self.recorder = recorder
        self.inits = 0
        self.raise_on = raise_on or set()

    def init(self):
        self.inits += 1
        return MockEngine(self.recorder, self.raise_on)


class VoiceTests(unittest.TestCase):
    def test_engine_initialized_once_and_sequential(self):
        recorder: list = []
        mod = EngineModule(recorder)
        main.pyttsx3 = mod

        v = main.VoiceAlert(enabled=True, capacity=8)

        v.speak("a:WARNING", "first")
        v.speak("b:WARNING", "second")
        v.speak("c:WARNING", "third")

        # allow worker to process
        time.sleep(0.3)
        v.close()

        self.assertEqual(mod.inits, 1)
        self.assertEqual(recorder, ["first", "second", "third"]) 

    def test_cooldown_suppresses_repeats(self):
        recorder: list = []
        mod = EngineModule(recorder)
        main.pyttsx3 = mod
        v = main.VoiceAlert(enabled=True, capacity=8)

        v.speak("car:WARNING", "msg")
        # immediate repeat should be suppressed by cooldown
        v.speak("car:WARNING", "msg")

        time.sleep(0.2)
        v.close()

        self.assertEqual(len(recorder), 1)

    def test_disabled_voice_creates_no_engine(self):
        recorder: list = []
        mod = EngineModule(recorder)
        main.pyttsx3 = mod
        v = main.VoiceAlert(enabled=False)

        v.speak("x:WARNING", "nope")
        v.close()

        self.assertEqual(mod.inits, 0)

    def test_tts_exceptions_do_not_crash(self):
        recorder: list = []
        mod = EngineModule(recorder, raise_on={"bad"})
        main.pyttsx3 = mod
        v = main.VoiceAlert(enabled=True, capacity=8)

        v.speak("a:WARNING", "bad")
        v.speak("b:WARNING", "good")

        time.sleep(0.3)
        v.close()

        # even though 'bad' raises, 'good' should still be spoken
        self.assertIn("good", recorder)


if __name__ == "__main__":
    unittest.main()
