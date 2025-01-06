from model.model import process_data
from data.audio_data import text_to_speech, speech_to_text

def main():
    # Example data for various tasks
    text_data = ("Hello, this is a test sentence.", "path_to_audio_output.mp3")  # Text-to-Speech
    audio_data = "path_to_audio_file.mp3"  # Audio-to-Text (ASR)
    audio_waveform_data = "path_to_audio_file.wav"  # Audio Classification / Voice Activity Detection
    
    # Text-to-Speech Example
    print("Text-to-Speech Conversion:")
    result = process_data({"text": "Hello, this is an audio test.", "audio": "output_test.mp3"})
    print(result)

    # Audio-to-Text Example
    print("\nAudio-to-Text Conversion:")
    text = process_data(audio_data)
    print(f"Recognized Text: {text}")

    # Audio Classification or VAD Example
    print("\nAudio Classification or VAD:")
    result = process_data(audio_waveform_data)
    print(result)

if __name__ == "__main__":
    main()
