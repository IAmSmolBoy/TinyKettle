"""Client for a self-hosted Ollama server, with running conversation history."""
import requests
import json
import os
from datetime import datetime
from threading import Thread


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the key facts, preferences, decisions, and unresolved topics "
    "from the conversation below in 3-6 short bullet points. Write it as "
    "notes for your own future reference, not a recap addressed to the user. "
    "Skip small talk and pleasantries - only keep information worth "
    "remembering for later conversations.\n\n"
    "Conversation:\n{transcript}"
)


class OllamaChat:
    def __init__(self, host, model, system_prompt=None, temperature=0.7,
                 keep_history=True, max_history_turns=12, memory_file="memory.json",
                 max_memories=50, summary_prompt=DEFAULT_SUMMARY_PROMPT):
        self.host = host.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.keep_history = keep_history
        self.max_history_turns = max_history_turns
        self.memory_file = memory_file
        self.max_memories = max_memories
        self.summary_prompt = summary_prompt
        self.history = []  # list of {"role": "user"/"assistant", "content": str}

    # ---------- Memory Helper ----------

    @staticmethod
    def _format_transcript(history):
        lines = []
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)
        

    def summarize(self) -> str:
        """Sends conversation history to the LLM and returns a summary string.

        This is a standalone /api/generate call (not appended to any live
        chat history), so it never pollutes an ongoing conversation.
        `history` is a list of {"role": "user"/"assistant", "content": str}.
        """
        transcript = self._format_transcript(self.history)
        prompt = self.summary_prompt_template.format(transcript=transcript)

        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def load_memories(self):
        """Returns the list of stored memories: [{timestamp, summary, ...}, ...]."""
        if not os.path.exists(self.memory_file):
            return []
        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_memories(self, memories):
        with open(self.memory_file, "w") as f:
            json.dump(memories, f, indent=2)

    def add_memory(self, summary, metadata=None):
        """Appends a summary to the memory store, trimming to max_memories."""
        if not summary:
            return
        
        memories = self.load_memories()
        
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
        }
        
        if metadata:
            entry["metadata"] = metadata
            
        memories.append(entry)
        if len(memories) > self.max_memories:
            memories = memories[-self.max_memories:]
        self._save_memories(memories)

    def clear_memories(self):
        self._save_memories([])

    # ---------- Memory ----------

    def summarize_and_store(self, metadata=None):
        """One-shot: summarize the given history and persist it.

        Returns the summary text.
        """
        summary = self.summarize()
        self.add_memory(summary, metadata=metadata)
        return summary

    # ---------- Chat Helper ----------

    def reset(self):
        self.history = []
    
    def setup_chat(self):
        
        msgs = []
        if self.system_prompt and len(self.system_prompt):
            msgs.append({"role": "system", "content": self.system_prompt})
        
        return msgs
    
    def parse_req(self, messages, stream_callback=None):

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": self.temperature},
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        full_reply = []
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                full_reply.append(piece)
                if stream_callback:
                    stream_callback(piece)
            if chunk.get("done"):
                break
            
        return "".join(full_reply).strip()

    # ---------- Chat ----------

    def ask(self, user_text, stream_callback=None):
        """Sends user_text plus history to Ollama, returns the full reply text.

        If stream_callback is given, it's called with each text chunk as it arrives.
        """
        messages = self.setup_chat()
            
        if self.keep_history:
            messages.extend(self.history)
            
        messages.append({"role": "user", "content": user_text})

        reply_text = self.parse_req(messages, stream_callback)

        if self.keep_history:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply_text})
            
            if len(self.history) >= self.max_history_turns * 2:
                summary = self.summarize_and_store()
                decoded = json.loads(summary)
                messages = self.setup_chat()
                messages.append({"role": "system", "content": "Conversation history: " + decoded["summary"]})
                self.history = messages
                print(summary, " <- summary")
            # self._trim_history()

        return reply_text
