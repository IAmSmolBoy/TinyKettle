# TinyKettle: An Offline Voice Assistant (Ollama + Whisper + Piper)

TinyKettle is a Python application that allows users to interact with a voice assistant using speech recognition and text-to-speech technologies. The assistant processes voice commands and provides responses using a language model, creating a seamless conversational experience.

```
Mic → VAD auto-stop → faster-whisper (STT) → Ollama (LLM) → Add memories → Piper (TTS) → Speaker
```

# Capabilities

- Offline voice assistant pipeline: mic → STT (Whisper) → LLM (Ollama) → TTS (Piper) → speaker
- Push-to-talk support via keyboard_listener (PTT)
- MicRecorder: VAD, silence detection, max-turn limits, configurable sample rate and VAD aggressiveness
- Transcriber: configurable model_size, device, compute_type, language
- OllamaChat: system/summarise prompts, temperature, conversation history, max history turns
- Speaker (TTS): local voice model + config, adjustable speaking_rate and volume
- play_audio helper to output synthesized audio to the speaker
- Text-only mode to test LLM without mic/TTS
- YAML-configurable via --config, default config.yaml

## 1. Prerequisites

- **Ollama** running locally with a model pulled:
  ```bash
  # install from https://ollama.com if you haven't already
  ollama pull llama3.1:8b
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

Add a free US-English voice into `voices/`. For other languages/accents,
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
You can also specify a custom configuration file:
```
python main.py --config my_config.yaml
```

## Components
- **main.py**: The main entry point of the application. It handles command-line arguments, loads the configuration, and runs either the voice or text loop based on user input.

- **components/audio_io.py**: Contains the `MicRecorder` class for recording audio from the microphone and the `play_audio` function for playing audio through the speakers.

- **components/llm.py**: Implements the `OllamaChat` class, which interfaces with the language model, managing conversation history and handling user queries.

- **components/ptt.py**: Includes the `keyboard_listener` function, which listens for push-to-talk events to activate the microphone for recording.

- **components/stt.py**: Defines the `Transcriber` class, which converts recorded audio into text using speech-to-text technology.

- **components/tts.py**: Contains the `Speaker` class, which synthesizes text into speech using text-to-speech technology.

## Project Structure
```
TinyKettle
├── main.py                # Main entry point for the application
├── components             # Directory containing the core components
│   ├── audio_io.py        # Audio input/output functionalities
│   ├── llm.py             # Language model interface
│   ├── ptt.py             # Push-to-talk functionality
│   ├── stt.py             # Speech-to-text conversion
│   └── tts.py             # Text-to-speech synthesis
├── config.yaml            # Configuration file for the project
├── requirements.txt       # Python dependencies
├── .gitignore             # Files to ignore in version control
└── README.md              # Project documentation
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