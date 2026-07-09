"""Speech-to-text using faster-whisper (fully offline)."""
import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int16", language=None):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio_int16):
        """audio_int16: 1-D numpy int16 array at `sample_rate` Hz."""
        audio_float = audio_int16.astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            audio_float,
            language=self.language,
            vad_filter=False,  # we already did VAD during recording
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
