"""
Offline voice assistant: mic -> Whisper -> Ollama -> Piper -> speaker.

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
from components.tts import Speaker
from components.ptt import keyboard_listener


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_components(cfg):
    transcriber = Transcriber(
        model_size=cfg["stt"]["model_size"],
        device=cfg["stt"]["device"],
        compute_type=cfg["stt"]["compute_type"],
        language=cfg["stt"]["language"],
    )
    chat = OllamaChat(
        host=cfg["ollama"]["host"],
        model=cfg["ollama"]["model"],
        system_prompt=cfg["ollama"]["system_prompt"],
        summary_prompt=cfg["ollama"]["summarise_prompt"],
        temperature=cfg["ollama"]["temperature"],
        keep_history=cfg["ollama"]["keep_history"],
        max_history_turns=cfg["ollama"]["max_history_turns"],
    )
    speaker = Speaker(
        voice_model_path=cfg["tts"]["voice_model"],
        voice_config_path=cfg["tts"]["voice_config"],
        speaking_rate=cfg["tts"]["speaking_rate"],
        volume=cfg["tts"]["volume"]
    )
    recorder = MicRecorder(
        sample_rate=cfg["audio"]["sample_rate"],
        vad_aggressiveness=cfg["audio"]["vad_aggressiveness"],
        silence_duration_sec=cfg["audio"]["silence_duration_sec"],
        max_turn_seconds=cfg["audio"]["max_turn_seconds"],
    )
    return recorder, transcriber, chat, speaker

def print_stack(e):
        
    # 1. Print the standard full error message (traceback)
    traceback.print_exc()
    
    # 2. Extract just the line number as an integer
    tb = e.__traceback__
    while tb.tb_next:
        tb = tb.tb_next

def start_listening(recorder: MicRecorder, transcriber: Transcriber, chat: OllamaChat, speaker: Speaker):
    
    try:

        audio = recorder.record_utterance()
        if audio is None:
            return

        user_text = transcriber.transcribe(audio)
        if not user_text:
            return
        print(f"You: {user_text}")

        reply_text = chat.ask(user_text)
        print(f"Assistant: {reply_text}\n")

        wav_audio, sr = speaker.synthesize(reply_text)
        play_audio(wav_audio, sr)
        
    except Exception as e:
        
        print_stack(e)
        
        with open("history.json", "w") as f:
            f.write(str(chat.history))
            
        exit(0)


def run_voice_loop(cfg):
    recorder, transcriber, chat, speaker = build_components(cfg)
    print(f"Ready. Model: {cfg['ollama']['model']} | esc to quit.\n")
    
    keyboard_listener(
        ptt_config=cfg["ptt"],
        callback=lambda : start_listening(recorder, transcriber, chat, speaker)
    )


def run_text_loop(cfg):
    """Text-only mode: no mic/TTS needed, useful for testing the Ollama connection."""
    
    chat = OllamaChat(
        host=cfg["ollama"]["host"],
        model=cfg["ollama"]["model"],
        system_prompt=cfg["ollama"]["system_prompt"],
        temperature=cfg["ollama"]["temperature"],
        keep_history=cfg["ollama"]["keep_history"],
        max_history_turns=cfg["ollama"]["max_history_turns"],
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
