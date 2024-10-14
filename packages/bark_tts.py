from bark import generate_audio,  preload_models, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
#from IPython.display import Audio
import numpy as np
import sounddevice as sd

preload_models()

def play_audio(audio_array, sample_rate=24000):
    """Play audio using sounddevice"""
    sd.play(audio_array, samplerate=sample_rate)
    sd.wait()  # Wait until the audio finishes playing

def speak(text):
    try:
        # Generate audio from text
        audio_array = generate_audio(text)
        breakpoint()
        # Play the generated audio
        play_audio(np.array(audio_array))
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")