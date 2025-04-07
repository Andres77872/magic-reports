from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import requests
import streamlit as st
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

from const import prompt_query_build
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
        model='@01/gpt-4o-2024-11-20' if openai_api_key else '@05/google/gemini-2.0-flash-001',
        private_key=openai_api_key if openai_api_key else None,
        base_url='https://llm.arz.ai'
    )

    prev_chat_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
    chat_query = ModelChat()
    chat_query.add_user_message(prompt_query_build.replace('{prev_chat}', prev_chat_context))

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
                    images_data.append((item['id'], item['page'], item['title'], item['url'], img_encoded))

    chat = ModelChat(
        system=(
            "You are a helpful assistant tasked with creating a detailed yet concise report. "
            "Your response must use information provided in paper page images when needed. "
            "Clearly cite each source within markdown in the format:\n\n"
            "> [Title of Paper](URL), page: X\n\n"
            "Always explicitly reference the Paper ID, title, page number, and URL provided."
        )
    )

    for paper_id, page_num, paper_title, paper_url, image_data in images_data:
        chat.add_user_message(
            content=(
                f"Use information from this paper page as necessary. "
                f"When citing this source, reference clearly as:\n"
                f"Paper ID: {paper_id}, Title: '{paper_title}', Page: {page_num}, URL: {paper_url}"
            ),
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
