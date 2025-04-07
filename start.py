import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from magic_llm import MagicLLM
from magic_llm.model import ModelChat


def scale_image(image: Image.Image, new_height: int = 1536) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height))


def fetch_image(i, new_height: int = 1536):
    image_url = i['page_image']
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        resized_image = scale_image(image, new_height)

        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        print(f"Successfully fetched and resized image from: {image_url}",
              f"(Size after encoding: {len(encoded_image) / 1024 / 1024:.2f} MB)")

        return (i['id'], i['page'], i['title'], i['url'], encoded_image)

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch image from {image_url}. Error: {e}")
        return None
    except Exception as e:
        print(f"Error processing image from {image_url}. Error: {e}")
        return None


with st.sidebar:
    openai_api_key = st.text_input("magic-llm API Key", key="chatbot_api_key", type="password")
    "[Colpali retrieval API](https://llm.arz.ai/docs#/data%20sources/colpali_rag_colpali_arxiv_post)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Colpali for Arxiv")
st.caption("A POC for the use of Colpali to embed and index the Arxiv papers, this is a basic demo to make RAG and "
           "write a simple report about a topic, the topic will be the user query and will be regenerated to "
           "generate 3-5 queries and search on colpali index, this is a POC and will be improved in the future. "
           "Only tested for one turn chat not for a conversation. The colpali index maintained by myself.")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    client = MagicLLM(
        engine='openai',
        model='@01/gpt-4o-2024-11-20' if openai_api_key else '@05/google/gemini-2.0-flash-001',
        private_key=openai_api_key if openai_api_key else None,
        base_url='https://llm.arz.ai'
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    images = []

    chat_q = ModelChat()

    prev_chat = [
        f"{x['role']}: {x['content']}"
        for x in st.session_state.messages[-6:]
    ]

    prev_chat = "\n".join(prev_chat)
    print(prev_chat)

    chat_q.add_user_message(f"""
You are assisting in generating relevant search queries for a Retrieval-Augmented Generation (RAG) system based on the user's last message in the conversation. The user's latest message is enclosed within "<user_query>" tags below:

<user_query>
{prev_chat}
</user_query>

Your task is to:

1. Analyze the user's query carefully to identify:
   - Main topics and intent
   - Specific details or context implied

2. Generate between 1 to 3 concise and varied search queries ONLY IF ADDITIONAL INFORMATION IS REQUIRED. These queries should:
   - Cover different angles of the user's intended question or request.
   - Include relevant synonyms, alternative phrasing, or supplemental context needed to retrieve valuable information for answering the query.

3. DO NOT generate queries for:
   - Simple factual questions (e.g., "What's 2+2?")
   - General greetings or irrelevant inputs ("Hi", "thanks").

4. Output your response strictly as a JSON array of strings. Never provide context, explanation, or extra text. If no additional information or queries are necessary, respond instantly with an empty JSON array.

Examples:

User query: "Differences between electric and diesel engines?"
Output:
["electric vs diesel engine comparison", "benefits of electric engines over diesel", "performance differences electric vs diesel engine", "environmental impact electric vs diesel engines"]

User query: "Thanks for your help!"
Output:
[]

Now, based strictly on the provided user's last query, generate the appropriate JSON array of queries, or an empty array if not required.
""".strip())
    resp = client.llm.generate(chat_q).content
    print(resp)
    resp = resp.strip().strip('`').lstrip('json').strip()
    print()
    queries = json.loads(resp)
    print(queries)
    query_result = []
    query_uniq = set()
    for q in queries:
        print(q)
        url = 'https://llm.arz.ai/rag/colpali/arxiv'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {
            'query': q,
            'limit': 4
        }

        qr = requests.post(url, headers=headers, data=data).json()
        for i in qr['data']:
            if i['page_image'] not in query_uniq:
                query_uniq.add(i['page_image'])
                query_result.append(i)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_image = {executor.submit(fetch_image, i): i for i in query_result}
        for future in as_completed(future_to_image):
            result = future.result()
            if result:
                images.append(result)

    chat = ModelChat(
        system=(
            "You are a helpful assistant tasked with creating a detailed yet concise report. "
            "Your response must use information provided in paper page images when needed. "
            "Clearly cite each source within markdown in the format:\n\n"
            "> [Title of Paper](URL), page: X\n\n"
            "Always explicitly reference the Paper ID, title, page number, and URL provided."
        )
    )

    for paper_id, page_num, paper_title, paper_url, image_data in images:
        chat.add_user_message(
            content=(
                f"Use information from this paper page as necessary. "
                f"When citing this source, reference clearly as:\n"
                f"Paper ID: {paper_id}, Title: '{paper_title}', Page: {page_num}, URL: {paper_url}"
            ),
            image=image_data,
            media_type='image/png'
        )
    # chat.add_user_message(
    #     'Just respond if you have the response on the pages and cite with table number, image, chart, topic, subtopic, etc. '
    #     'Also if you found images, figures, diagrams, Etc related with the user Query describe with the reference:\n\n Query:' + query_str)

    chat.messages.extend(st.session_state.messages)

    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream the response
        for chunk in client.llm.stream_generate(chat):
            if hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")

        # Update the final response without the cursor
        response_placeholder.markdown(full_response)

    # Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
