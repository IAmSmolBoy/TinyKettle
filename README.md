# Offline Voice Assistant (Ollama + Whisper + Piper)

Hold a spoken conversation with a locally-run LLM. No cloud services, no API keys.

```
Mic → VAD auto-stop → faster-whisper (STT) → Ollama (LLM) → Piper (TTS) → Speaker
```

## 1. Prerequisites

- **Ollama** running locally with a model pulled:
  ```bash
  # install from https://ollama.com if you haven't already
  ollama pull llama3
  ollama serve   # if not already running as a service
  ```
- **Python 3.9+**
- A working microphone and speakers/headphones.
- System audio libraries (Linux):
  ```bash
  sudo apt-get install libportaudio2 portaudio19-dev
  ```

## 2. Install Python dependencies

```bash
cd voice_assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Get a Piper voice

```bash
chmod +x download_voice.sh
./download_voice.sh
```
This grabs a free US-English voice into `voices/`. For other languages/accents,
browse [Piper's voice list](https://github.com/rhasspy/piper/blob/master/VOICES.md)
and update `tts.voice_model` / `tts.voice_config` in `config.yaml` to match.

## 4. Configure

Open `config.yaml` and adjust:
- `ollama.model` — must match a model you've pulled (`ollama list` to check)
- `ollama.system_prompt` — this is your assistant's **personality**. Rewrite it
  to change tone, expertise, or character.
- `stt.model_size` — `tiny`/`base` are fast on CPU; `small`/`medium` are more
  accurate but slower. Use `device: cuda` if you have an NVIDIA GPU.
- `audio.silence_duration_sec` — how long you can pause before it decides
  you're done talking. Lower = snappier, higher = more patient with pauses.

## 5. Run it

```bash
python main.py
```

Speak after "Listening..." appears — it auto-detects when you stop talking
and processes your turn. Press Ctrl+C to quit.

To test just the LLM connection without a mic (e.g. over SSH), use:
```bash
python main.py --text-only
```

## Customizing further

| Want to...                          | Edit...                                    |
|--------------------------------------|---------------------------------------------|
| Change the assistant's personality    | `ollama.system_prompt` in `config.yaml`     |
| Swap LLM models                       | `ollama.model` in `config.yaml`             |
| Use a different voice/accent          | Download a new Piper voice, update `tts.*`  |
| Make it more/less "trigger-happy"     | `audio.vad_aggressiveness` (0-3)            |
| Run on GPU                            | `stt.device: cuda`, `compute_type: float16` |
| Point at a remote Ollama server       | `ollama.host` (e.g. `http://192.168.1.5:11434`) |

## Troubleshooting

- **No module named 'sounddevice' errors on Linux**: install `portaudio19-dev`
  first (see step 1), then reinstall `sounddevice`.
- **Ollama connection refused**: confirm `ollama serve` is running and
  `ollama.host` in the config matches its address/port (default `11434`).
- **It cuts me off mid-sentence**: raise `audio.silence_duration_sec`.
- **It never stops listening**: lower `audio.vad_aggressiveness` isn't the
  fix — try increasing `vad_aggressiveness` (more aggressive = filters non-
  speech noise better) or checking your mic input level isn't picking up
  background hum as speech.
