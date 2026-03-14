import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import wave
from datetime import datetime

import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# -------------------------
# SETTINGS
# -------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds

# Mikrofon index
INPUT_DEVICE = 2

# Whisper model: tiny, base, small, medium, large-v3 ...
WHISPER_MODEL_SIZE = "small"

# Piper voice model dosyalari
VOICE_MODEL_NAME = "tr_TR-fahrettin-medium.onnx"
VOICE_CONFIG_NAME = "tr_TR-fahrettin-medium.onnx.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "transcripts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_wav_info(filepath):
    with wave.open(filepath, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        duration = frames / float(rate)

    return {
        "frames": frames,
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": sampwidth,
        "duration_seconds": round(duration, 3),
    }


def transcribe_audio(audio_path):
    print("\nSTT modeli yükleniyor...")

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

    print("Transkripsiyon başlıyor...")
    segments, info = model.transcribe(
        audio_path,
        language="tr"
    )

    full_text = ""
    for segment in segments:
        full_text += segment.text.strip() + " "

    return full_text.strip(), info


def synthesize_speech(text, output_path):
    voice_model_path = os.path.join(BASE_DIR, VOICE_MODEL_NAME)
    voice_config_path = os.path.join(BASE_DIR, VOICE_CONFIG_NAME)

    if not os.path.exists(voice_model_path):
        print(f"TTS voice model bulunamadı: {voice_model_path}")
        return False

    if not os.path.exists(voice_config_path):
        print(f"TTS voice config bulunamadı: {voice_config_path}")
        return False

    if not text.strip():
        print("TTS için metin boş geldi, ses üretimi atlandı.")
        return False

    print("\nTTS modeli yükleniyor...")
    voice = PiperVoice.load(voice_model_path, voice_config_path)

    print("Metin sese çevriliyor...")
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    return True


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_wav_file = os.path.join(RECORDINGS_DIR, f"input_{timestamp}.wav")
    output_wav_file = os.path.join(OUTPUTS_DIR, f"output_{timestamp}.wav")
    metadata_file = os.path.join(METADATA_DIR, f"meta_{timestamp}.json")
    transcript_file = os.path.join(TRANSCRIPTS_DIR, f"transcript_{timestamp}.txt")

    print("Kayıt başlıyor...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=INPUT_DEVICE
    )
    sd.wait()

    write(input_wav_file, SAMPLE_RATE, audio)
    print(f"WAV kaydedildi: {input_wav_file}")

    wav_info = get_wav_info(input_wav_file)

    with open(input_wav_file, "rb") as f:
        audio_bytes = f.read()

    transcript_text, info = transcribe_audio(input_wav_file)

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    tts_created = synthesize_speech(transcript_text, output_wav_file)

    metadata = {
        "timestamp": timestamp,
        "input_wav_file": input_wav_file,
        "output_wav_file": output_wav_file if tts_created else None,
        "transcript_file": transcript_file,
        "device_index": INPUT_DEVICE,
        "binary_size_bytes": len(audio_bytes),
        "language": "tr",
        "stt_model": f"faster-whisper-{WHISPER_MODEL_SIZE}",
        "tts_model": VOICE_MODEL_NAME if tts_created else None,
        "detected_language_probability": getattr(info, "language_probability", None),
        "transcript_text": transcript_text,
        **wav_info,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n--- TRANSCRIPT ---")
    print(transcript_text if transcript_text else "[Boş çıktı]")
    print("------------------")

    if tts_created:
        print(f"TTS kaydedildi: {output_wav_file}")
    else:
        print("TTS çıktısı oluşturulamadı.")

    print(f"Transcript kaydedildi: {transcript_file}")
    print(f"Metadata kaydedildi: {metadata_file}")


if __name__ == "__main__":
    main()