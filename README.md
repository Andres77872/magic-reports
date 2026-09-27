# Colpali-Arxiv AI Chatbot

Streamlit chat that answers questions about Arxiv papers using [Colpali](https://huggingface.co/vidore/colpali-v1.3) retrieval and a vision language model through [MagicLLM](https://github.com/Andres77872/magic-llm).

Enter a topic. The app rewrites it into search queries, retrieves matching paper pages from the Colpali Arxiv index, sends those page images to the model, and streams a cited markdown report.

The Colpali index is maintained separately. This app is the chat interface on top of it.

## How a question is answered

1. The selected model rewrites the latest messages into a JSON list of search queries.
2. Each query is sent to `https://llm.arz.ai/rag/colpali/arxiv`.
3. Unique page images are downloaded, resized, and encoded as PNG.
4. Each page image is attached to a final chat request, along with recent conversation history and the original question.
5. The model streams a report. Sources and token usage appear under the last answer.

You can regenerate the last answer, edit the Jinja2 prompt templates, and change generation and search settings from the sidebar.

## Requirements

- Python 3.11
- A MagicLLM API key when the selected model requires one (models whose id starts with `@01`)

## Setup

```bash
pip install -r requirements.txt
streamlit run start.py
```

The app listens on port 8501.

In a dev container, dependencies install on content update and Streamlit starts on attach. The preview opens on port 8501.

## Configuration

Sidebar controls:

| Setting | Default | Role |
| --- | --- | --- |
| Model | Gemini 2.0 Flash | Model used for query rewrite and the final report |
| Temperature | 0.75 | Randomness of generation |
| Top-P | 0.95 | Nucleus sampling |
| Max new tokens | 4096 | Response length cap |
| Presence / frequency / repetition | 0.0 / 0.0 / 1.0 | Repetition penalties |
| Rewrites | 5 | How many search queries to generate |
| Results per query | 4 | Colpali hits kept per query |
| Image height | 1536 | Resize height of page images before they are sent to the model |

Models are defined in `model_choices` in `const.py`. Requests go to `https://llm.arz.ai` with the OpenAI-compatible MagicLLM client.

Prompt templates live in `const.py` and can be edited in the sidebar:

- **Search query generation** uses `{{ prev_chat }}` and `{{ query_rewrite_count }}`.
- **System prompt** sets citation and evidence rules for the report.
- **Colpali context prompt** can use paper fields such as `{{ id }}`, `{{ title }}`, `{{ page }}`, `{{ url }}`, `{{ authors }}`, `{{ abstract }}`, `{{ date }}`, `{{ doi }}`, `{{ version }}`, and `{{ page_image }}`.

## Project layout

| File | Role |
| --- | --- |
| `start.py` | Streamlit UI and the retrieval-to-answer pipeline |
| `const.py` | Prompt templates, app description, and model list |
| `utils.py` | Colpali search and image fetch/encode |
| `requirements.txt` | Pinned Python dependencies |
| `.devcontainer/devcontainer.json` | Dev container that installs deps and runs Streamlit |
| `.streamlit/config.toml` | Commented Streamlit theme options |

## Links

- [Colpali retrieval API](https://llm.arz.ai/docs#/data%20sources/colpali_rag_colpali_arxiv_post)
- [Colpali model](https://huggingface.co/vidore/colpali-v1.3)
- [MagicLLM docs](https://llm.arz.ai/docs)
