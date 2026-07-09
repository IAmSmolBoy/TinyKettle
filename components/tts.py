"""Text-to-speech using Piper (fully offline, natural-sounding voices)."""
import io
import wave
import numpy as np
from piper.voice import PiperVoice
from piper.config import SynthesisConfig


class Speaker:
    def __init__(self, voice_model_path, voice_config_path, speaking_rate=1.0, volume=1.0):
        self.voice = PiperVoice.load(voice_model_path, config_path=voice_config_path)
        self.sample_rate = self.voice.config.sample_rate
        
        self.syn_config = SynthesisConfig(
            length_scale=1/speaking_rate,       # Speeds up the voice (original length / speaking_rate)
            volume=volume,              # Full volume (0.0 to 1.0)
            normalize_audio=True     # Maximizes volume consistency
        )

    def synthesize(self, text):
        """Returns (int16 numpy array, sample_rate)."""

        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            self.voice.synthesize_wav(
                text,
                wav_file,
                syn_config=self.syn_config
            )
        buf.seek(0)
        
        with wave.open(buf, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
        
        
        audio = np.frombuffer(frames, dtype=np.int16)
        return audio, self.sample_rate
    
