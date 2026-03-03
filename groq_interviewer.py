"""
groq_interviewer.py
──────────────────
Wraps the Groq client and manages per-session conversation history
for the AI Voice Interview mode.

Role-aware: uses candidate_name + job_role to generate relevant questions.
Max 15 questions, then auto-evaluates. Also evaluates on 'exit' command.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MAX_QUESTIONS = 15


def _build_system_prompt(candidate_name: str, job_role: str) -> str:
    return (
        f"You are a STRICT professional interviewer conducting a real job interview.\n\n"

        f"CANDIDATE INFO:\n"
        f"- Name: {candidate_name}\n"
        f"- Applying for: {job_role}\n\n"

        f"ROLE LOCK:\n"
        f"- You are ALWAYS the interviewer. The user is ALWAYS the candidate.\n"
        f"- NEVER switch roles, explain concepts, or answer questions.\n"
        f"- If the candidate asks you something, reply: "
        f"'This is an interview. Please answer the question.'\n\n"

        f"QUESTION PLAN — Ask exactly in this order, ONE question at a time:\n\n"

        f"PHASE 1 — General HR (first 4-5 questions):\n"
        f"  1. Tell me about yourself.\n"
        f"  2. What are your key strengths and weaknesses?\n"
        f"  3. Why do you want to work as a {job_role}?\n"
        f"  4. Where do you see yourself in 5 years?\n"
        f"  5. How do you handle pressure or tight deadlines?\n\n"

        f"PHASE 2 — Technical Questions specific to '{job_role}' (next 8-10 questions):\n"
        f"  - Ask deep, role-specific technical questions relevant to the skills, "
        f"tools, and responsibilities of a {job_role}.\n"
        f"  - Adapt follow-up questions based on what the candidate mentions "
        f"(e.g., if they mention Python, ask about Python deeply).\n"
        f"  - Cover: core concepts, problem-solving scenarios, tools/technologies, "
        f"system design (if applicable), past project experience.\n\n"

        f"STRICT RULES:\n"
        f"1. Ask ONE question at a time. Wait for the answer before asking the next.\n"
        f"2. Keep questions concise and professional.\n"
        f"3. Do NOT provide hints, answers, or corrections.\n"
        f"4. Do NOT give lengthy speeches.\n"
        f"5. Maximum {MAX_QUESTIONS} questions total. After question {MAX_QUESTIONS}, "
        f"IMMEDIATELY deliver the final evaluation.\n\n"

        f"FINAL EVALUATION FORMAT (use this EXACTLY when the interview ends):\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 INTERVIEW EVALUATION\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Candidate: {candidate_name}\n"
        f"Role Applied: {job_role}\n"
        f"Questions Asked: [N]\n\n"
        f"Overall Score: [X] / 10\n\n"
        f"✅ Strengths:\n"
        f"- [point 1]\n"
        f"- [point 2]\n\n"
        f"⚠️ Weak Areas:\n"
        f"- [point 1]\n"
        f"- [point 2]\n\n"
        f"💡 Suggestions for Improvement:\n"
        f"- [point 1]\n"
        f"- [point 2]\n\n"
        f"🏆 Recommendation: [HIRE / CONSIDER / REJECT]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━"
    )


class GroqSession:
    """
    Holds the full conversation history for one candidate session.
    Call .chat(message) to get the next AI response.
    """

    def __init__(self, candidate_name: str, job_role: str = "Software Developer"):
        self.candidate_name = candidate_name
        self.job_role = job_role
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.history: list[dict] = [
            {"role": "system", "content": _build_system_prompt(candidate_name, job_role)}
        ]
        self.question_count: int = 0
        self.is_finished: bool = False

    def chat(self, user_message: str) -> str:
        """Send a candidate message and return the interviewer's reply."""

        # Check if user wants to exit early
        if user_message.strip().lower() == "exit":
            self.is_finished = True

        self.history.append({"role": "user", "content": user_message})

        # If max questions reached and not already flagged, inject a finish signal
        if self.question_count >= MAX_QUESTIONS and not self.is_finished:
            self.is_finished = True
            self.history.append({
                "role": "user",
                "content": "[SYSTEM: Maximum questions reached. Deliver the final evaluation now.]"
            })

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=self.history,
            temperature=0.4,
            max_completion_tokens=1024,
            stream=True,
        )

        full_response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta

        self.history.append({"role": "assistant", "content": full_response})

        # Count questions asked by AI (ends with '?')
        if full_response.strip().endswith("?"):
            self.question_count += 1

        # Auto-finish after delivering evaluation
        if self.question_count >= MAX_QUESTIONS:
            self.is_finished = True

        return full_response

    def get_transcript(self) -> list[dict]:
        """Return the conversation minus the system message."""
        return [m for m in self.history if m["role"] != "system"]
