import os
import time
from dotenv import load_dotenv

# Importing a speech to text module
import assemblyai as aai

# Testing with different API/local LLMs
from packages.sales_chatbot import SalesChatbot
from packages.llama_chatbot import LlamaChatbot

# Testing with different API/local TTS models
from packages.elevenlabs_tts import speak  # Update the class name if different
import packages.bark_tts as bark_tts
import packages.cartesia_tts as cartesia_tts


load_dotenv()


class VoiceBot:

    def __init__(self):
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.transcriber = None
        self.chatbot = LlamaChatbot()  # SalesChatbot()
        self.transcription_active = True

    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=500,
        )

        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(microphone_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)
        return

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if self.transcription_active:

            if isinstance(transcript, aai.RealtimeFinalTranscript):
                print("[User]: " + transcript.text, end="\n")
                self.respond(transcript.text)

            else:
                print("[User]: " + transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        return

    def on_close(self):
        return

    def respond(self, transcript):
        self.transcription_active = False

        # self.stop_transcription()

        # generate response from OpenAI
        response = self.chatbot.generate_response(transcript)

        print("[Bot]: ", response)

        # speak response using ElevenLabs
        # speak(response)
        bark_tts.speak(response)

        self.transcription_active = True

        # self.start_transcription()


voice_bot = VoiceBot()
voice_bot.start_transcription()
