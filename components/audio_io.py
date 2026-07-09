"""Microphone capture with voice-activity-based auto-stop, plus playback."""
import collections
import queue
import sys
import numpy as np
import sounddevice as sd
import webrtcvad


class MicRecorder:
    """Records from the microphone and auto-stops after trailing silence."""

    def __init__(self, sample_rate=16000, frame_ms=30, vad_aggressiveness=2,
                 silence_duration_sec=0.9, max_turn_seconds=30):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_frames_needed = int(silence_duration_sec * 1000 / frame_ms)
        self.max_frames = int(max_turn_seconds * 1000 / frame_ms)
        self._q = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self._q.put(indata.copy())

    def record_utterance(self, require_speech_first=True):
        """Blocks until the user finishes speaking (or stays silent the whole time).

        Returns int16 numpy array of audio, or None if nothing was said.
        """
        self._q = queue.Queue()
        ring = collections.deque(maxlen=self.silence_frames_needed)
        voiced_frames = []
        triggered = False
        frame_count = 0

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                             blocksize=self.frame_samples, callback=self._callback):
            print("Listening...")
            while frame_count < self.max_frames:
                block = self._q.get()
                frame_count += 1
                pcm_bytes = block.tobytes()
                try:
                    is_speech = self.vad.is_speech(pcm_bytes, self.sample_rate)
                except Exception:
                    is_speech = False

                if not triggered:
                    ring.append((block, is_speech))
                    num_voiced = len([f for f, s in ring if s])
                    if num_voiced > 0.6 * ring.maxlen and is_speech:
                        triggered = True
                        for f, _ in ring:
                            voiced_frames.append(f)
                        ring.clear()
                else:
                    voiced_frames.append(block)
                    ring.append((block, is_speech))
                    num_unvoiced = len([f for f, s in ring if not s])
                    if num_unvoiced >= ring.maxlen and len(ring) == ring.maxlen:
                        break

        if not voiced_frames or (require_speech_first and not triggered):
            return None

        audio = np.concatenate(voiced_frames, axis=0).flatten()
        return audio


def play_audio(audio_bytes_or_array, sample_rate):
    """Plays back int16 PCM audio (numpy array or raw bytes)."""
    audio_np = audio_bytes_or_array.astype(np.float32) / 32768.0
    
    sd.play(audio_np, sample_rate)
    # sd.wait()
