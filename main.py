"""
Offline voice assistant: MicRecorder -> Whisper -> Ollama -> Piper -> play_audio.

Usage:
    python main.py
    python main.py --config my_config.yaml
    python main.py --text-only     # skip mic/TTS, type/read instead (for testing)
"""
import argparse
import yaml
import traceback

from components.audio_io import MicRecorder, play_audio
from components.stt import Transcriber
from components.llm import OllamaChat
from components.tts import Voice
from components.ptt import keyboard_listener


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_components(cfg):
    transcriber = Transcriber(
        **cfg["stt"],
    )
    chat = OllamaChat(
        **cfg["ollama"]
    )
    voice = Voice(
        **cfg["tts"]
    )
    recorder = MicRecorder(
        **cfg["mic"]
    )
    return recorder, transcriber, chat, voice

def print_stack(e):
        
    # 1. Print the standard full error message (traceback)
    traceback.print_exc()
    
    # 2. Extract just the line number as an integer
    tb = e.__traceback__
    while tb.tb_next:
        tb = tb.tb_next

def start_listening(recorder: MicRecorder, transcriber: Transcriber, chat: OllamaChat, voice: Voice):
    
    try:

        audio = recorder.record_utterance()
        if audio is None:
            return

        user_text = transcriber.transcribe(audio)
        if not user_text:
            return
        print(f"You: {user_text}")

        print(f"Assistant:", end=" ")
        reply_text = chat.ask(user_text)
        
        wav_audio, sr = voice.synthesize(chat._remove_header(reply_text))
        play_audio(wav_audio, sr)
        
    except Exception as e:
        
        print_stack(e)
        exit(0)


def run_voice_loop(cfg):
    recorder, transcriber, chat, voice = build_components(cfg)
    print(f"Ready. Model: {cfg['ollama']['model']} | {cfg['ptt']["exit_hotkey"]} to quit.\n")
    
    keyboard_listener(
        ptt_config=cfg["ptt"],
        callback=lambda : start_listening(recorder, transcriber, chat, voice)
    )


def run_text_loop(cfg):
    """Text-only mode: no mic/TTS needed, useful for testing the Ollama connection."""
    
    chat = OllamaChat(
        **cfg["ollama"]
    )
    print(f"Text-only mode. Model: {cfg['ollama']['model']} | Ctrl+C to quit.\n")
    
    while True:
        try:
            user_text = input("You: ").strip()
            if not user_text:
                continue
            
            reply_text = chat.ask(user_text)
            print(f"Assistant: {reply_text}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--text-only", action="store_true",
                         help="Skip mic/TTS - type messages, test the LLM connection only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.text_only:
        run_text_loop(cfg)
    else:
        run_voice_loop(cfg)
