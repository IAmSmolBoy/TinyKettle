from pynput import keyboard
from components.audio_io import MicRecorder, play_audio
from components.stt import Transcriber
import yaml




def _start_listening():

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    recorder = MicRecorder(
        sample_rate=cfg["audio"]["sample_rate"],
        vad_aggressiveness=cfg["audio"]["vad_aggressiveness"],
        silence_duration_sec=cfg["audio"]["silence_duration_sec"],
        max_turn_seconds=cfg["audio"]["max_turn_seconds"],
    )
    transcriber = Transcriber(
        model_size=cfg["stt"]["model_size"],
        device=cfg["stt"]["device"],
        compute_type=cfg["stt"]["compute_type"],
        language=cfg["stt"]["language"],
    )
    
    audio = recorder.record_utterance()
            
    if audio is None:
        return

    user_text = transcriber.transcribe(audio, cfg["audio"]["sample_rate"])
    
    if not user_text:
        return
    
    print(f"You: {user_text}")
    
def ptt(key, hotkey, exit_hotkey, callback=None):
    try:
        if key.char == hotkey:
            callback() if callback else None
    
    except AttributeError:
        if exit_hotkey == "esc" and key == keyboard.Key.esc:
            return False
        if exit_hotkey == "end" and key == keyboard.Key.end:
            return False
        
def keyboard_listener(ptt_config, callback=None):
    # Collect events until released
    with keyboard.Listener(
        on_press=lambda key: ptt(
            key,
            ptt_config["hotkey"],
            ptt_config["exit_hotkey"],
            callback=callback if callback else _start_listening
        ),
    ) as listener:
        listener.join()
    




if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    keyboard_listener(cfg["ptt"])