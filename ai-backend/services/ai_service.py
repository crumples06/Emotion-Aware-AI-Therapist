import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import Retriever

# Load .env variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY is not set in environment variables!")

# Initialize Groq client
client = Groq(api_key=api_key)

# System prompt (same as your Groq version)
SYSTEM_PROMPT = """
You are a supportive, non-clinical mental health assistant for college students.

Your goals:
1. Listen empathetically and ask reflective questions.
2. Build on prior conversations and guide students through their emotions.
3. Provide stress management, mindfulness, and healthy coping strategies.
4. Encourage journaling, exercise, rest, and social connection when relevant.
5. If a student shows serious distress (extreme hopelessness, self-harm, or suicidal thoughts), gently suggest professional counselling.
6. Never act like a doctor or prescribe medication.
7. Always validate the student's feelings before offering suggestions.

Your style:
- Warm, conversational, and supportive - like a peer counselor, not a generic chatbot.
- Acknowledge feelings first, then gently explore with short open-ended questions.
- Use grounding techniques, affirmations, and coping strategies.
- Maintain context from earlier in the conversation.
- Keep responses natural and 3-5 sentences long.
"""""


class GroqService:
    """
    Groq-powered AI service for therapy responses
    """

    def __init__(self):
        self.sessions = {}
        self.model_name = "llama-3.1-8b-instant"
        self.max_messages = 20

        print("Initializing RAG retriever...")
        self.retriever = Retriever()
        print("RAG retriever ready.")

    def get_or_create_history(self, session_id: str) -> list:
        # If this session doesnt exist yet, create it
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self.sessions[session_id]

    def trim_history(self, history: list) -> list:
        # Always keep the System prompt
        system_prompt = history[0]

        # Keep only the last max_messages from the conversation history
        recent_messages = history[1:][-self.max_messages:]
        return [system_prompt] + recent_messages

    async def test_connection(self):
        """Test Groq API connection"""
        try:
            # Simple test prompt
            response = await self.generate_therapy_response("Hello, this is a test.")
            print(f"✅ Groq API connection successful with model: {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Groq connection test failed: {str(e)}")
            return False

    async def generate_therapy_response(
            self,
            user_message: str,
            emotion: str = "neutral",
            session_id: str = "default"
    ) -> str:
        try:
            # Step 1: Get this user's history
            history = self.get_or_create_history(session_id)

            # Step 2: Retrieve relevant context from knowledge base
            # Run in executor because retrieve() is synchronous and blocking
            # This prevents the server from freezing while searching Pinecone
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                lambda: self.retriever.retrieve(user_message)
            )
            context = self.retriever.format_context(chunks)

            # Step 3: Build emotion and context aware message
            enriched_message = user_message

            if emotion and emotion != "neutral":
                enriched_message = f"[The user appears to be feeling {emotion}.] {enriched_message}"

            if context:
                enriched_message = f"{context}\n\nUser message: {enriched_message}"

            # Step 4: Add to history
            history.append({"role": "user", "content": enriched_message})

            # Step 5: Trim history
            history = self.trim_history(history)
            self.sessions[session_id] = history

            # Step 6: Call Groq
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    max_tokens=300,
                    temperature=0.7
                )
            )

            # Step 7: Extract response
            bot_message = response.choices[0].message.content

            # Step 8: Save to history
            self.sessions[session_id].append(
                {"role": "assistant", "content": bot_message}
            )

            return bot_message.strip()

        except Exception as e:
            return f"I'm having trouble responding right now. Please try again. (Error: {str(e)})"

    def clear_session(self, session_id: str):
        # Clear a session's history - call when user logs out
        if session_id in self.sessions:
            del self.sessions[session_id]

# Optional: test from terminal
if __name__ == "__main__":
    import asyncio

    service = GroqService()
    print("Groq Health-Bot is running. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Take care! Ending the session now.")
            break
        reply = asyncio.run(service.generate_therapy_response(user_input))
        print("Bot:", reply)
