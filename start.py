import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
from jinja2 import Template
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

from const import prompt_query_build, prompt_system_llm, prompt_colpali_content, helper_prompt_configuration_jinja2
from utils import fetch_and_encode_image

st.set_page_config(
    page_title="Colpali-Arxiv AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar UI enhancements
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png",
             use_container_width=True)
    st.markdown("## 🔑 API Configuration")
    openai_api_key = st.text_input(
        "Magic-LLM API Key (optional)",
        type="password"
    )

    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    🌐 [Colpali Retrieval API](https://llm.arz.ai/docs#/data%20sources/colpali_rag_colpali_arxiv_post)  

    🤗 [Embedding Model](https://huggingface.co/vidore/colpali-v1.3)  

    📂 [View Source Code](https://github.com/Andres77872/magic-reports)
    """)

st.title("💬 Colpali-Arxiv Chat: AI-Powered Retrieval and Reporting")
st.markdown(
    "This demo showcases the capabilities of Colpali for embedding, indexing, and generating retrieval-augmented "
    "responses from Arxiv research papers. Enter a topic query, and the system will reformulate your query into "
    "3-5 targeted searches using the Colpali index, subsequently generating a concise summary report. "
    "Currently, this is a proof-of-concept focusing on single-turn chats, with conversational enhancements planned for the future. "
    "The Colpali index is independently maintained."
)

st.markdown("## 📝 Prompt Configuration (Jinja2 syntax)")

with st.expander("📖 Show detailed Jinja2 keys reference", expanded=False):
    st.markdown(helper_prompt_configuration_jinja2)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.expander("#### 🔎 Search Query Prompt", expanded=False):
        user_prompt_template = st.text_area(
            "Prompt to rewrite user query into targeted retrieval queries",
            value=prompt_query_build,
            height=280,
            help="Use variables like `{{ prev_chat }}` to dynamically inject recent conversation context."
        )

with col2:
    with st.expander("#### 🤖 LLM System Prompt", expanded=False):
        system_prompt_template = st.text_area(
            "System-level instructions for the AI assistant",
            value=prompt_system_llm,
            height=280,
            help="Provide overarching instructions for assistant behavior. Example: 'You are an assistant specialized in summarizing scientific papers from Arxiv.'"
        )

with col3:
    with st.expander("#### 📑 Colpali Context Prompt", expanded=False):
        colpali_prompt_template = st.text_area(
            "Prompt guiding how paper content is presented to the LLM",
            value=prompt_colpali_content,
            height=280,
            help=(
                "You can use the following keys from the paper context dictionary:\n\n"
                "- `{{ page }}`: Page number of the retrieved content.\n"
                "- `{{ id }}`: Arxiv paper identifier.\n"
                "- `{{ doi }}`: DOI of paper (if available, can be null).\n"
                "- `{{ date }}`: Publication date.\n"
                "- `{{ title }}`: Paper title.\n"
                "- `{{ authors }}`: Paper authors.\n"
                "- `{{ abstract }}`: Paper abstract.\n"
                "- `{{ url }}`: URL to the original Arxiv paper.\n"
                "- `{{ version }}`: Paper version number.\n"
                "- `{{ page_image }}`: URL of the retrieved page image.\n\n"
                "These keys can enrich your prompt templates to effectively present paper information to the LLM."
            )
        )

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

prompt = st.chat_input("What's your question about Arxiv papers?")

if prompt:
    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    client = MagicLLM(
        engine='openai',
        model='@01/gpt-4o-2024-11-20' if openai_api_key else '@10/accounts/fireworks/models/llama4-maverick-instruct-basic',
        private_key=openai_api_key if openai_api_key else None,
        base_url='https://llm.arz.ai'
    )

    prev_chat_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
    chat_query = ModelChat()

    prompt_template = Template(user_prompt_template)
    rendered_prompt = prompt_template.render(prev_chat=prev_chat_context)

    chat_query.add_user_message(rendered_prompt)

    with st.spinner("Generating related queries...", show_time=True):
        queries_json = client.llm.generate(chat_query).content.strip().lstrip("```json").rstrip("```")
        queries = json.loads(queries_json)

    # fetch paper data
    all_paper_data = []
    unique_images = set()

    progress_text = st.empty()  # placeholder for displaying current query
    progress_bar = st.progress(0)

    for idx, query in enumerate(queries):
        progress_text.markdown(f"🔍 **Processing Query ({idx + 1}/{len(queries)}):** `{query}`")

        # Send request
        res = requests.post(
            'https://llm.arz.ai/rag/colpali/arxiv',
            data={'query': query, 'limit': 4}
        ).json()

        for paper in res.get('data', []):
            if paper['page_image'] not in unique_images:
                all_paper_data.append(paper)
                unique_images.add(paper['page_image'])

        progress_bar.progress((idx + 1) / len(queries))

    progress_bar.empty()
    progress_text.empty()

    images_data = []
    with st.spinner("Fetching paper images...", show_time=True):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_and_encode_image, item['page_image']) for item in all_paper_data]
            for future, item in zip(as_completed(futures), all_paper_data):
                img_encoded = future.result()
                if img_encoded:
                    images_data.append((item, img_encoded))

    chat = ModelChat(system=system_prompt_template)

    for content_dict, image_data in images_data:
        prompt_template = Template(colpali_prompt_template)
        rendered_prompt = prompt_template.render(**content_dict)
        chat.add_user_message(
            content=rendered_prompt,
            image=image_data,
            media_type='image/png'
        )

    # Stream response with spinner visible while waiting for first token
    with st.chat_message("assistant"):
        placeholder_response = st.empty()
        full_response = ""

        # Create a container for the spinner
        spinner_container = st.empty()
        with spinner_container:
            with st.spinner("✨ Generating response, please wait..."):
                response_generator = client.llm.stream_generate(chat)
                first_token_received = False

                # Iterate through response generator
                for chunk in response_generator:
                    if hasattr(chunk, 'choices'):
                        content = chunk.choices[0].delta.content
                        if content:
                            if not first_token_received:
                                first_token_received = True
                                spinner_container.empty()  # Remove the spinner when first token arrives
                            full_response += content
                            placeholder_response.markdown(full_response + "▌")

        # Finalize full response without cursor
        placeholder_response.markdown(full_response)

    # Update session messages
    st.session_state.messages.append({"role": "assistant", "content": full_response})
