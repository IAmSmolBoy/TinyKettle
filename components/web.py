"""
web_search.py - Tavily-powered web search for LLM grounding.
"""
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()


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
                    "description": "The search query. Be specific.",
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


class WebSearchError(Exception):
    pass


class WebSearch:
    """Client for the Tavily Search API."""

    SEARCH_ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key=None, timeout=15):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise WebSearchError(
                "No API key found. Set TAVILY_API_KEY as an env var, or pass "
                "api_key= directly. Sign up free at https://app.tavily.com"
            )
            
        self.timeout = timeout

    def search(self, query, num_results=5, depth="basic", topic="general",
               include_answer=False, include_domains=None, exclude_domains=None,
               days=None):
        """Run a Tavily search and return structured results.

        query:           what to search for
        num_results:     how many results to return (1-10)
        depth:           "basic" (1 credit, fast snippets) or
                         "advanced" (2 credits, full extracted page content)
        topic:           "general" or "news" (biases results toward news articles)
        include_answer:  if True, Tavily also returns its own AI-generated
                         answer summarising the results (costs no extra credits)
        include_domains: list of domains to restrict results to, e.g. ["bbc.com"]
        exclude_domains: list of domains to exclude
        days:            restrict to results from the last N days (news topic only)

        Returns list of:
            {
                "title":   str,
                "url":     str,
                "content": str,   # snippet (basic) or full text (advanced)
                "score":   float, # Tavily's relevance score 0-1
                "answer":  str | None  # only present when include_answer=True
            }
        """
        # Construct payload with mandatory parameters
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(max(int(num_results), 1), 10),
            "search_depth": depth,
            "topic": topic,
            "include_answer": include_answer,
            "include_raw_content": False,
        }
        
        # Conditionally add optional parameters to payload
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if days:
            payload["days"] = days

        # Send POST request to Tavily API and handle errors
        resp = requests.post(self.SEARCH_ENDPOINT, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise WebSearchError(
                f"Tavily API error {resp.status_code}: {resp.text[:300]}"
            )
            
        # Parse response and extract results
        data = resp.json()
        answer = data.get("answer")  # present when include_answer=True
        results = []
        for item in data.get("results", []):
            result = {
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "content": item.get("content", "").replace("\n", " ").strip(),
                "score":   round(item.get("score", 0.0), 3),
            }
            if include_answer:
                result["answer"] = answer
            results.append(result)
        return results

    def search_formatted(self, query, **kwargs):
        """Same as search(), but returns a single text block ready to hand
        to an LLM context window.
        """
        # Get results from search method
        results = self.search("With the above results, " + query, **kwargs)
        
        # Handle no results case
        if not results:
            return f"No results found for '{query}'."
        
        # Format results into a single text block
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"\n{i}. {r['title']}\n"
                # f"   URL: {r['url']}\n"
                f"   {r['content']}"
            )
        return "\n".join(lines)

    def search_news(self, query, num_results=5, days=7):
        """Convenience wrapper: news-biased search limited to recent results."""
        return self.search(query, num_results=num_results, topic="news", days=days)


# ---------------------------------------------------------------------------
# Ollama tool-calling integration
# ---------------------------------------------------------------------------

def execute_tool_call(tool_call, search_client: WebSearch):
    """Runs a single Ollama tool_call dict against the search client.

    Returns the text to send back to the model as the tool result.
    """
    fn = tool_call.get("function", {})
    name = fn.get("name")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    if name != "web_search":
        return f"Error: unknown tool '{name}'"

    query = args.get("query", "")
    num = args.get("num_results", 5)
    depth = args.get("depth", "basic")
    topic = args.get("topic", "general")

    if not query:
        return "Error: no query provided."

    try:
        return search_client.search_formatted(
            query, num_results=num, depth=depth, topic=topic
        )
    except WebSearchError as e:
        return f"Search failed: {e}"


def ask_with_search(host, model, user_text, search_client, system_prompt=None,
                    history=None, temperature=0.7, max_tool_rounds=4, timeout=90):
    """Full Ollama round trip with Tavily search as a tool.

    Sends user_text to Ollama with the web_search tool available. Each
    time the model requests a search, Tavily is called and the results are
    fed back. This repeats until the model gives a final answer or
    max_tool_rounds is reached.

    host:            Ollama server, e.g. "http://localhost:11434"
    model:           model name — must support tool calling
                     (llama3.1, mistral-nemo, qwen2.5 all work)
    user_text:       the user's question
    search_client:   a WebSearch instance
    system_prompt:   optional system prompt override
    history:         optional list of prior {"role","content"} messages
    max_tool_rounds: safety cap on search→reply cycles
    timeout:         seconds before giving up on an Ollama request

    Returns the model's final reply as a string.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    
    for i in range(max_tool_rounds):
        resp = requests.post(
            f"{host.rstrip('/')}/api/chat",
            json={
                "model":   model,
                "messages": messages,
                "tools":   [SEARCH_TOOL_SCHEMA],
                "stream":  False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return message.get("content", "").strip()

        # Append assistant's tool-call turn, then execute each call and
        # feed results back as tool messages before the next round.
        messages.append(message)
        for call in tool_calls:
            result_text = execute_tool_call(call, search_client)
            messages.append({"role": "tool", "content": result_text})

    # Safety exit: model kept requesting tools past max_tool_rounds
    messages.append({
        "role":    "user",
        "content": "Please give your final answer now based on the search results above.",
    })
    resp = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json={
            "model":   model,
            "messages": messages,
            "stream":  False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Demo — run directly with: python web_search.py
# Requires TAVILY_API_KEY env var and Ollama running llama3.1 (or similar)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    search = WebSearch()

    # print("=== Direct basic search ===")
    # print(search.search_formatted("latest Ollama release", num_results=3))

    # print("\n=== Direct news search (last 3 days) ===")
    # for r in search.search_news("AI agent frameworks", num_results=3, days=3):
    #     print(f"  {r['title']} — {r['url']}")

    print("\n=== LLM-driven search (model picks when/what to search) ===")
    answer = ask_with_search(
        host="http://localhost:11434",
        model="llama3.1:8b",
        user_text="Tell me about the latest news regarding Donald Trump not regarding Iran",
        search_client=search,
        system_prompt=(
            "You are a concise assistant. Use the web_search tool for any "
            "question involving current events, versions, or recent changes."
        ),
    )
    print(answer)