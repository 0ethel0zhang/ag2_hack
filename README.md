This project leverages the **AG2 Beta** framework to build sophisticated multi-agent systems. The current implementation demonstrates a parallel web research pattern where a lead agent coordinates specialized researchers.

## Architecture & Technology Stack
- **Framework:** AG2 Beta (`autogen.beta`)
- **LLM Providers:** Primarily Google Gemini (configured via `GeminiConfig`) and OpenAI (configured via `OpenAIConfig`).
- **Tools:** 
  - `TavilyClient`: Used for web searching (`tavily_search`) and content extraction (`fetch_url`).
- **Core Pattern:** 
  - **Lead Agent:** Decomposes complex questions into sub-tasks and synthesizes final reports.
  - **Researcher Agents:** Execute specific search and retrieval tasks in parallel.
  - **LaneRouter:** Provides live, interleaved progress tracking for concurrent agent operations in the terminal.

## Setup & Configuration
1.  **Environment Variables:** Create a `.env` file with the following keys:
    - `TAVILY_API_KEY`: Required for web search and extraction.
    - `AG2_GEMINI_API_KEY`: API key for Gemini models.
    - `LLM_PROVIDER`: Set to `gemini` (default) or `openai`.
2.  **Dependencies:** Ensure `autogen`, `tavily`, and `python-dotenv` are installed.

## Building and Running
- **Run the Parallel Research Agent:**
  ```bash
  python main.py
  ```
- **Tests:** [TODO: Add testing framework and commands]

## Development Conventions
- **Asynchronous Execution:** The framework relies heavily on `asyncio`. Ensure all tool definitions and agent interactions are `async`.
- **Tool Definition:** Use the `@tool` decorator from `autogen.beta` for defining agent-accessible functions.
- **Progress Tracking:** Utilize the `LaneRouter` (or similar event-based systems) to provide visibility into multi-agent workflows.
- **Model Selection:** Prefer `gemini-2.5-pro` for reasoning/synthesis (Lead) and `gemini-2.5-flash` for high-volume tasks (Researchers).

## Sample Output
<img width="1001" height="882" alt="Screenshot 2026-05-03 at 2 48 24 PM" src="https://github.com/user-attachments/assets/16912b97-5772-4818-8158-4897d0c134fd" />
