"""Client for a self-hosted Ollama server, with running conversation history."""
import requests
import json
import re
from pathlib import Path
import uuid

from components.web import WebSearch, execute_tool_call


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the key facts, preferences, decisions, and unresolved topics "
    "from the conversation below in 3-6 short bullet points. Write it as "
    "notes for your own future reference, not a recap addressed to the user. "
    "Skip small talk and pleasantries - only keep information worth "
    "remembering for later conversations.\n\n"
    "Conversation:\n{transcript}"
)


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-world information on any topic. "
            "Use this whenever the user asks about something recent, a fact "
            "you are not certain about, or anything that may have changed "
            "since your training data was collected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query optimized for a search engine. Not for something you already can do.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10). Default 5.",
                },
                "depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": (
                        "Search depth. 'basic' returns snippets (fast, 1 credit). "
                        "'advanced' fetches and extracts full page content "
                        "(thorough, 2 credits). Use advanced for detailed questions."
                    ),
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": "Use 'news' to bias results toward recent news articles.",
                },
            },
            "required": ["query"],
        },
    },
}


class OllamaChat:
    def __init__(self, host, model, system_prompt=None, temperature=0.7,
                 keep_history=True, max_msgs=12, summary_prompt=DEFAULT_SUMMARY_PROMPT,
                 chats_folder="chats", max_tool_rounds= 4):
        self.host = host.rstrip("/")  # Remove trailing slash from host URL
        self.model = model  # Model to be used for chat generation
        self.system_prompt = system_prompt  # Optional system prompt for the chat
        self.temperature = temperature  # Temperature parameter for text generation
        self.keep_history = keep_history  # Whether to keep conversation history
        self.max_msgs = max_msgs  # Maximum number of turns to store in history
        self.summary_prompt = summary_prompt  # Prompt to use for summarization
        self.history = []  # List of {"role": "user"/"assistant", "content": str}
        self.chats_folder = chats_folder  # Folder to store chat files
        self.chat_title = ""  # Title of the current chat
        self.chat_filename = ""  # Filename to save the current chat
        self.search = WebSearch()  # WebSearch client instance
        self.max_tool_rounds = max_tool_rounds  # Safety cap on search→reply cycles

        # Define the folder path
        folder_path = Path(chats_folder)

        # Create folder if it doesn't exist (including any missing parent folders)
        folder_path.mkdir(parents=True, exist_ok=True)
    
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server returned status code {response.status_code}.")
        
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ConnectionError("Ollama server is not running. Please start Ollama first.") from e
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama server returned an error: {e}")

        # Get the contents of the folder
        chat_files = [chat_mem_file for chat_mem_file in folder_path.iterdir()]
        
        print("-" * 75, "Chats", "-" * 75)
        
        # Iterate through to extract all chats
        self.chats = {}
        for path in chat_files:
    
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)  # Load JSON data from the file
                title = data.get("title", "")  # Retrieve the title or use an empty string if not present
                
                if title in self.chats:  # If chat has unique title
                    
                    # Incrementally append a number to avoid duplicate titles
                    same_name_count = len([ key for key in self.chats.keys() if title in key ])
                    title += f" {same_name_count + 1}"
                    
                # Store filename and messages of the chat with title as a key
                self.chats[title] = {
                    "filename": path,  
                    "messages": data.get("messages", [])
                }

        if self.chats != {}:  # If there are any chats stored
            while True:
                i = 0  # Initialize counter for printing options
                
                for title in self.chats:  # Loop through each chat title
                    
                    print(f"{i + 1}. {title}")  # Print option number and title
                    i += 1
                
                try:
                    # Get user's selection
                    new_chat = int(input(f"{i + 1}. New chat\nPlease indicate the chat you would like to load: "))
                    
                except:
                    print("Selection unrecognised. Please try again.")  # Handle invalid selection
                    continue
                
                if new_chat > i + 1 or new_chat < 1:
                    print("Selection unrecognised. Please try again.")  # Handle invalid selection
                    continue
                    
                if new_chat == i + 1:
                    break  # Exit loop if 'new chat' is selected

                self.chat_title = list(self.chats.keys())[new_chat - 1]  # Set the selected chat title
                
                # Retrieve chat data based on selected title
                chat_data =  self.chats[self.chat_title]
                
                self.history = chat_data["messages"]  # Add chat's messages to history
                self.chat_filename = chat_data["filename"]  # Store filename of the selected chat
                break

        for msg in self.history:  # Get each message in the chat history
            match msg["role"]:
                case "user":
                    print("You: ", end="")  # Print 'You:' if the message is from a user
                case "assistant":
                    print("Assistant: ", end="")  # Print 'assistant:' if the message is from an assistant
                case _:
                    print("System:", end="")  # Default to printing as a System message
            print(msg["content"])  # Print the content of the message


    # ---------- Chat Helper ----------

    @staticmethod
    def _remove_header(msg):
        
        # Removes headers from the message that starts and ends with one or two asterisks.
        return re.sub(r"\*{1,2}(.*?)\*{1,2}", r'\1', msg)

    @staticmethod
    def _extract_json(msg):
        
        # Extracts the first JSON object or array from string.
        obj = re.search(r"(\{.*\})", msg, re.DOTALL)
        
        # Get regex result and convert to dict
        return json.loads(obj.group(1).replace("*", ""))
    
    @staticmethod
    def _format_transcript(history):
        '''Format history into "{role}: {message}" for each message'''
        
        lines = []
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
            
        # Join all messages with \n
        return "\n".join(lines)
        


    def setup_chat(self, include_hist=False):
        
        # Includes system prompt into messages if provided.
        msgs = []
        if self.system_prompt and len(self.system_prompt):
            msgs.append({"role": "system", "content": self.system_prompt})

        if include_hist and self.keep_history:
            msgs.extend(self.history)  # Add past conversation history to current session
        
        return msgs
    
    def chat(self, settings):

        if not settings:
            raise Exception("Json not provided.")
    
        # Send request to the chat model
        resp = requests.post(
            f"{self.host}/api/chat",
            json.dumps(settings),
            stream=True,
            timeout=120,
        )
        
        # Raises HTTPError, if one occurred.
        resp.raise_for_status()

        full_reply, tool_calls = [], []
        
        # Iterate through each line of the response.
        for line in resp.iter_lines():
            
            # Skip empty lines
            if not line:
                continue
            
            # Convert line to json
            chunk = json.loads(line)
            message = chunk.get("message", {})
            
            # Collect and print model's responses
            if message.get("content"):
                full_reply.append(message["content"])
                print(message["content"], end="", flush=True)
                
            # Collect tool calls made by the model
            if message.get("tool_calls"):
                tool_calls.extend(message["tool_calls"])
                print(message["tool_calls"])
                    
            if chunk.get("done"):
                print()
                break
            
        return "".join(full_reply).strip(), tool_calls
    
    def no_search_prompt(self, messages: list):
        
        return self.chat({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        })
    
    def search_prompt(self, messages: list):

        # print({
        #     "model": self.model,
        #     "messages": messages,
        #     "tools":   [SEARCH_TOOL_SCHEMA],
        #     "stream": True,
        #     "options": {"temperature": self.temperature},
        # })
        
        result_text = ""
            
        # Initialize WebSearch client instance
        search = WebSearch()

        # Loop to interact with the chat model up to max_tool_rounds times
        for i in range(self.max_tool_rounds):
            
            full_reply, tool_calls = self.chat({
                "model": self.model,
                "messages": messages,
                "tools":   [SEARCH_TOOL_SCHEMA],
                "stream": True,
                "options": {"temperature": self.temperature},
            })
            
            # Handle tool calls if any
            if not tool_calls:
                return "".join(full_reply).strip(), result_text
            
            for call in tool_calls:
                result_text = execute_tool_call(call, search)
                messages.append({"role": "tool", "content": result_text})
                print("web:", result_text, "\n")

        # Final request to get the final answer based on search results
        messages.append({
            "role":    "user",
            "content": "Please give your final answer now based on the search results above.",
        })
        
        full_reply, _ = self.chat({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        })
        
        return "".join(full_reply).strip(), result_text
    
    def summarise_all(self):
        """
        Summarize the conversation history in chunks and update the history.
        
        Returns:
            None
        """
        
        summaries = []
        i = 0
        while i <= len(self.history) - self.max_msgs:
            
            # Extract a chunk of messages for summarization
            transcript = self._format_transcript(self.history[i:i + self.max_msgs])
            
            # Format the prompt with the extracted transcript
            prompt = self.summary_prompt.format(transcript=transcript)

            # Send a POST request to the chat API
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=120,
            )
            resp.raise_for_status()
            
            # Extract and append the summary to the summaries list
            summary = resp.json().get("response", {})
            summaries.append({"role": "assistant", "content": summary})
            
            # Print summary
            print("Summary:\n", summary)
            
            i += self.max_msgs
            
        self.history = summaries  # Update current history with summarized version

    # ---------- Chat ----------

    def ask(self, user_text):
        """Sends user_text plus history to Ollama, returns the full reply text.

        If stream_callback is given, it's called with each text chunk as it arrives.
        """
        
        messages = self.setup_chat(include_hist=True)  # Initialize chat message list
        messages.append({"role": "user", "content": user_text})  # Append the new user message to the conversation
        
        web = ""
        while True:
            web = input("Web search? (N): ").lower()
            if not web or web in ["n", "no"]:
                reply_text = self.no_search_prompt(messages)  # Send messages to Ollama and get the response
                break
            elif web in ["y", "yes"]:
                messages.insert(-1, {
                    "role": "system",
                    "content": "You are a concise assistant. Use the web_search tool for any "
                    "question involving current events, versions, or recent changes."
                })
                reply_text = self.search_prompt(messages)
                break
            
            print("Selection unknown, please retry.")

        # If chat has no title, ask llm to add a header.
        if self.chat_title == "":
            # Get new chat title
            resp = requests.post(
                f"{self.host}/api/generate",
                json.dumps({
                    "model": self.model,
                    "prompt": f"Output a unique, suitable header about the request with NO other words. Headers NOT available: {", ".join(list(self.chats.keys()))}. The request: " + user_text,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                }),
                timeout=120,
            )
            
            # Raises HTTPError, if one occurred.
            resp.raise_for_status()
            
            new_title = self._remove_header(resp.json().get("response", ""))
            self.chat_title = new_title
                    
        if self.keep_history:
            self.history = messages[1:]  # Add new messages to history
            self.history.append({
                "role": "assistant",
                "content": reply_text[0]
            })
            
            if len(self.history) >= self.max_msgs:  # If history is too long
                
                print("Chat history getting long, summarising...")
                self.summarise_all()  # Summarize and store the conversation history

            if self.chat_filename == "":  # If chat filename is not set
                self.chat_filename = self.chats_folder + f"/{len(self.chats)}. " + uuid.uuid4().hex + ".json"  # Generate a new UUID for the chat file
            
            with open(self.chat_filename, "w") as f:
                f.write(json.dumps({
                    "title": self.chat_title,
                    "messages": self.history
                }))  # Save the chat history to a JSON file

        return reply_text  # Return the full reply text