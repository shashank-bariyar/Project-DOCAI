#step1: setup audio recorder (ffmpeg & portaudio)
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(messages)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    simplified func to record audio from the microphone and save it as an mp3 file.

    Args:
        file_path (str): path to save recorded audio file
        timeout (int): max time to wait for a phrese to start.
        phrase_time_limit (int): max time for phrase to be recorded (in seconds).
        
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("ADJUSTING FOR AMBIENT NOISE...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("START SPEAKING NOW...")
            
            #RECORD AUDIO
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("RECORDING COMPLETE.")
            
            #CONVERT RECORDED AUDIO TO MP3 FILE.
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")
            
    except Exception as e:
        logging.error(f"An error occured: {e}")

audio_filepath="patient_voice_test.mp3"
#record_audio(file_path=audio_filepath)

#step2: setup speech to text STT model for transcription
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
stt_model="whisper-large-v3-turbo"

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    client=Groq(api_key=GROQ_API_KEY)
    audio_file=open(audio_filepath,"rb")
    transcription=client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
        )
    return transcription.text
