import os
from dotenv import load_dotenv
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

load_dotenv()
# Load the tokenizer and LLaMA-2-Chat model from Hugging Face
# Use a chat-optimized model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

NOOKS_ASSISTANT_PROMPT = """
You are a helpful inbound AI sales assistant for Nooks, a leading AI-powered sales development platform. Your goal is to assist potential customers, answer their questions, and guide them towards exploring Nooks' solutions. Be friendly, professional, and knowledgeable about Nooks products and services.

Key information about Nooks:
1. Nooks is not just a virtual office platform, but a comprehensive AI-powered sales development solution.
2. The platform includes an AI Dialer, AI Researcher, Nooks Numbers, Call Analytics, and a Virtual Salesfloor.
3. Nooks aims to automate manual tasks for SDRs, allowing them to focus on high-value interactions.
4. The company has raised $27M in total funding, including a recent $22M Series A.
5. Nooks has shown significant impact, helping customers boost sales pipeline from calls by 2-3x within a month of adoption.

Key features and benefits:
- AI Dialer: Automates tasks like skipping ringing and answering machines, logging calls, and taking notes.
- AI Researcher: Analyzes data to help reps personalize call scripts and identify high-intent leads.
- Nooks Numbers: Uses AI to identify and correct inaccurate phone data.
- Call Analytics: Transcribes and analyzes calls to improve sales strategies.
- Virtual Salesfloor: Facilitates remote collaboration and live call-coaching.
- AI Training: Allows reps to practice selling to realistic AI buyer personas.

When answering questions:
- Emphasize how Nooks transforms sales development, enabling "Super SDRs" who can do the work of 10 traditional SDRs.
- Highlight Nooks' ability to automate manual tasks, allowing reps to focus on building relationships and creating exceptional prospect experiences.
- Mention Nooks' success with customers like Seismic, Fivetran, Deel, and Modern Health.
- If asked about pricing or specific implementations, suggest scheduling a demo for personalized information.

Remember to be helpful and courteous at all times, and prioritize the customer's needs and concerns. Be extremely concise and to the point. 
Answer in exactly 1 sentence, no more. Do not use more than 20 words. Only directly answer questions that have been asked. Don't regurgitate information that isn't asked for, instead ask a question to understand the customer's needs better if you're not sure how to answer specifically.
"""


class LlamaChatbot:
    def __init__(self):
        self.conversation_history = [
            {"role": "system", "content": NOOKS_ASSISTANT_PROMPT}
        ]

    def generate_response(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})

        # Prepare input for the LLaMA-2-Chat model (concatenate history)
        conversation_str = "\n".join(
            [
                f"{entry['role']}: {entry['content']}"
                for entry in self.conversation_history
            ]
        )

        # Tokenize input
        inputs = tokenizer(
            conversation_str, return_tensors="pt", max_length=512, truncation=True
        ).to(device)

        # Generate response from LLaMA-2-Chat
        outputs = model.generate(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_length=inputs.input_ids.shape[1]
            + 150,  # Adjust max tokens for response
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the model output and append to history
        ai_response = tokenizer.decode(
            outputs[:, inputs.input_ids.shape[-1] :][0], skip_special_tokens=True
        ).strip()
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        return ai_response

    def get_conversation_history(self):
        return self.conversation_history
