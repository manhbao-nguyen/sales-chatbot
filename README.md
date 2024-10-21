# AI Sales Assistant Chatbot

## Project Goal

This project aims to create a realistic, low-latency chatbot that functions as an AI sales assistant for Nooks, an AI-powered sales development platform.
The chatbot responds when the user falls silent for some time, simulating a natural conversation flow.

https://www.loom.com/share/ce9d6443c02e4ae19e5ca3994b58e3df

The current implementation is relatively slow to respond the the user - the goal is to make it faster.

## Current Implementation

The system consists of three main components:

1. Speech-to-text (STT) using AssemblyAI's hosted API for real-time transcription
2. A sales chatbot powered by OpenAI's GPT-4 model (note - not gpt4o which is faster but not as accurate in some cases)
3. Text-to-speech (TTS) using ElevenLabs for voice output

The chatbot listens to user input, transcribes it in real-time, and generates a response when the user stops speaking. The AI's response is then converted to speech and played back to the user.

## Task

Assume that you are not allowed to modify the services used (you must use Assembly's hosted model for STT, OpenAI's GPT-4 for the chatbot, and ElevenLabs with this voice setting for TTS).
How would you modify the code to make the chatbot lower latency & respond faster?

### Evaluation Criteria

Your solution will be evaluated based on:

1. Reduction in overall latency
2. Maintenance of conversation quality and realism (i.e the chatbot doesn't interrupt the human speaker while they're in the middle of speaking)
3. Code quality and clarity of explanation

## Getting Started

1. Review the existing code in `main.py`, `lib/sales_chatbot.py`, and `lib/elevenlabs_tts.py`
2. Install the requirements by running `pip install -r requirements.txt` (or use a virtual environment if you prefer)
3. Set your OpenAI, AssemblyAI, and ElevenLabs API keys in `.env` - you should have received them via email.
4. Run the current implementation to understand its behavior by running `python3 main.py`
5. Begin your optimization process. Document your changes and reasoning in this README.md file when done.

## Poetry Setup

If you're getting stuck with installation issues, we offer an alternative Poetry-based installation method.

1. Install [Poetry](https://python-poetry.org/docs/#installing-with-pipx)
2. Install all requirements by running `poetry install`
3. Run the current implementation by running `poetry run python3 main.py`

Good luck!

## Bonus

Right now a lot of the latency comes from external services (TTS, LLM inference, STT). An easy way to reduce latency would be to use local models.
For example you could use:

- Nemo ASR for STT
- Llama for the chatbot
- bark or tortoise for TTS
  Try building a version of this chatbot that is local-only and see what speedup you achieve!


# Initiatives (Manh-Bao Nguyen, 10.20.2024)

Overall process: Upon trying to make the base code run, I had some initial issues (limits of API keys provided for ElevenLabs and OpenAI), which made me resort to free models for the chatbot LLM and the text-to-speech generator. I present the different steps hereafter. 

**1. Running the initial code: bottlenecks**
The API key provided for elevenlabs had too few credits to perform initial tests.
The API key provided for OpenAI was not valid anymore.

Note: I managed to run the audio -> text and text -> text parts of the pipeline a few times with a personal OpenAI key that I soon replaced with a local LLM to limit costs.
ElevenLabs never worked due to limits of calling the API with the key provided, but replacing it by [bark](https://github.com/suno-ai/bark) worked smoothly. 
However, the text-to-speech inference with bark was very slow because I was running the code on my local MAC CPU. 
I have access to a cluster with GPUs via ssh and tried to setup the repository, but it turned out that outputing the audio on my mac from the ssh instance in real-time was challenging (very slow, and I do not have the sudo permission on this cluster to install certain audio dependencies). 

In terms of latency improvements, it was challenging to test them because the local pipeline with local LLM  and bark was too slow to have decent realistic outputs. However, here are the ideas explored: 

a. Trying different local LLMs and text-to-speech models

b. Preventing the re-instantiation of the transcriber after the bot answer, i.e. pausing the transcription instead of deleting the transcriber after the end of the user intervention.

c. Generating the text2text answer of the LLM on the fly instead of waiting for the entire input user sequence to arrive.

**2. Local LLM: Llama**
In the `packages.llama_chatbot.py`, the provided code using the OpenAI API was adapted for the use of a small Llama-chat-7b model. 
Yet, the model was still to slow to run one inference on a MAC CPU.

**3. Local TTS: bark, cartesia**
In `packages.bark_tts.py`, the TTS `speak` method was adapted to locally use the open-source bark. Using `tortoise` required GPU power. 
One (personally) exciting initiative was to also experiment with the new TTS model based on State-Space Models, Cartesia ([reference](https://github.com/cartesia-ai/cartesia-python)). Indeed, the start-up claim their TTS model has the best latency and quality (human benchmark) - notably, the emotion / rhythm / voice granularity are highly modular wit this model, which is key to have the least uncanny human-computer interaction in the final app. The related code is provided in `packages.cartesia_tts.py`


**4. Algorithmic improvements**
In the provided code, at the beginning of the response generation by the LLM with the collected text, the transcriber was stopped and reinstantiated after the generated audio. This is not necessary since we can just manually pause the transcription on data when we generate the text/audio in the background. The code is accordingly modified in `main.py`. 


Lastly, an incremental text generation was considered, with the intuition that we should not wait for the whole user input to start generating an answer with the LLM. The latter could generate a partial answer from a first input chunk (eg after a small pause in the user speech) and confirm/modify the first answer with the upcoming chunks. This is not too hard to engineer, but calling the LLM a second time to confirm its initial answer with the latest chunks would not have improved much on the latency with hindsight...


**5. Conclusion and future ideas**
It was unfortunate to face initial issues with the provided API keys. 
However, it was interesting to see a concrete  speech to speech pipeline and to play around with state-of-the-art local substitute components, as well as providing a slight improvement circumventing the transcriber re-instantiation 
Finally, a rigorous evaluation of the latency improvement could have been provided - but as the global pipeline proved impossible to run on a local MAC without GPU / valid API keys, this is left for future work.


Time spent on this assignment: 2h15