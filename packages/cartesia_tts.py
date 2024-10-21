from cartesia import Cartesia
import pyaudio
import os
from dotenv import load_dotenv

load_dotenv()

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
voice_name = "Barbershop Man"
voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
voice = client.voices.get(id=voice_id)


model_id = "sonic-english"

# reference: https://docs.cartesia.ai/reference/api-reference/rest/stream-speech-server-sent-events
output_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 44100,
}

p = pyaudio.PyAudio()
rate = 44100

stream = None


def speak(text):
    """Convert text to speech and play it using Cartesia."""
    voice = client.voices.get(id=voice_id)  # Fetch the voice details

    stream = None
    try:
        # Streaming audio using Server-Sent Events (SSE)
        for output in client.tts.sse(
            model_id=model_id,
            transcript=text,
            voice_embedding=voice["embedding"],
            stream=True,
            output_format=output_format,
        ):
            buffer = output["audio"]

            # Initialize the stream only once
            if not stream:
                stream = p.open(
                    format=pyaudio.paFloat32, channels=1, rate=rate, output=True
                )

            # Write the audio data to the stream
            stream.write(buffer)

    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")

    finally:
        # Stop and close the stream and terminate PyAudio
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
