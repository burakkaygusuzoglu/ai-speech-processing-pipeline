# AI Speech Processing Pipeline

A simple AI-based speech processing pipeline built with open-source tools.

This project records speech from a microphone, converts speech into text using an AI speech recognition model, then converts the recognized text back into speech using a local neural text-to-speech engine.

---

## Features

* Record audio from microphone
* Save audio as WAV file
* Convert speech to text
* Save transcript as text file
* Convert text back to speech
* Save generated speech as WAV file
* Store metadata in JSON format
* Automatically play generated output audio

---

## Project Pipeline

Microphone Input
↓
WAV Recording
↓
Speech-to-Text
↓
Transcript Save
↓
Text-to-Speech
↓
Output WAV
↓
Metadata JSON

---

## Technologies Used

### Speech-to-Text

* faster-whisper
  Optimized implementation of OpenAI Whisper for fast speech recognition.

### Text-to-Speech

* Piper
  Local neural text-to-speech engine using ONNX voice models.

### Audio Recording

* sounddevice
  Microphone recording library.

### Audio File Handling

* scipy
  Used for WAV file writing.

### Audio Playback

* simpleaudio
  Used for automatic playback of generated audio.

### Metadata Handling

* JSON
  Stores recording information and transcript details.

---

## Open Source Libraries

* faster-whisper
* Piper
* sounddevice
* scipy
* simpleaudio

---

## Folder Structure

speech_ai_project/

├── app.py
├── recordings/
├── transcripts/
├── outputs/
├── metadata/
├── tr_TR-fahrettin-medium.onnx
└── tr_TR-fahrettin-medium.onnx.json

---

## Installation

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install faster-whisper
python -m pip install sounddevice scipy numpy
python -m pip install piper-tts
python -m pip install simpleaudio
```

---

## Voice Model

This project uses a Turkish Piper voice model.

Required files:

* tr_TR-fahrettin-medium.onnx
* tr_TR-fahrettin-medium.onnx.json

These files must be placed in the project root directory.

---

## Run

```bash
python app.py
```

---

## Output Files

### recordings/

Recorded microphone input WAV files

### transcripts/

Speech recognition text output

### outputs/

Generated speech output WAV files

### metadata/

JSON metadata for each recording

---

## Example Metadata

```json
{
  "timestamp": "...",
  "input_wav_file": "...",
  "output_wav_file": "...",
  "transcript_text": "...",
  "sample_rate": 16000
}
```

---

## Future Improvements

* Real-time speech recognition
* GPU acceleration
* GUI interface
* Database storage
* API integration

---

## Purpose

This project demonstrates a complete speech → text → speech pipeline using local open-source AI models.
