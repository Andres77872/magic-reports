import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from jinja2 import Template, TemplateError
from magic_llm import MagicLLM
from magic_llm.model import ModelChat
# Assuming these are defined elsewhere as before
from const import (prompt_query_build,
                   prompt_system_llm,
                   prompt_colpali_content,
                   helper_prompt_configuration_jinja2,
                   model_choices,
                   app_description)
from utils import fetch_and_encode_image, fetch_colpali_data

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Colpali-Arxiv AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Helper Functions ---
DEFAULT_MODEL_KEY = list(model_choices.keys())[0] # Get the first model key as default

# --- Sidebar ---
with st.sidebar:
    st.image(
        "https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png",
        use_container_width=True
    )
    st.markdown("## 🔑 API Credentials")
    openai_api_key = st.text_input(
        "Magic-LLM API Key (optional)",
        type="password",
        help="Enter your Magic-LLM API key if required by the selected model."
    )
    if not openai_api_key and model_choices[DEFAULT_MODEL_KEY].startswith('@01'): # Example check
         st.warning("An API key might be needed for the default or selected model.", icon="⚠️")

    st.divider() # Visual separator

    st.markdown("## ⚙️ Core Configuration")
    selected_model = st.selectbox(
        "🤖 Select Model",
        list(model_choices.keys()),
        index=0, # Default to the first model in the list
        help="Choose the Large Language Model for generating responses."
    )

    # --- Advanced Configuration Expander ---
    with st.expander("🛠️ Advanced Settings", expanded=False):

        st.markdown("#### LLM Generation Parameters")
        temperature = st.slider(
            "🌡️ Temperature", 0.0, 2.0, 1.0, 0.01,
            help="Controls randomness. Lower values are more deterministic."
        )
        top_p = st.slider(
            "🎯 Top-P", 0.0, 1.0, 1.0, 0.01,
            help="Nucleus sampling probability threshold."
        )
        max_tokens = st.number_input(
            "📝 Max New Tokens", 1, 8192, 4096, 1,
            help="Maximum length of the generated response."
        )
        presence_penalty = st.slider(
            "👤 Presence Penalty", -2.0, 2.0, 0.0, 0.01,
            help="Penalizes new tokens based on their appearance in the text so far."
        )
        frequency_penalty = st.slider(
            "🔄 Frequency Penalty", -2.0, 2.0, 0.0, 0.01,
            help="Penalizes new tokens based on their frequency in the text so far."
        )
        repetition_penalty = st.slider(
            "♻️ Repetition Penalty", 0.5, 2.0, 1.0, 0.01,
            help="Penalizes repetition (>1 less likely, <1 more likely)."
        )

        st.markdown("#### Colpali Search Parameters")
        query_rewrite_count = st.slider(
            "✍️ Query Rewrites", 1, 10, 5, 1,
            help="Number of search queries generated from your input."
        )
        result_count = st.slider(
            "📚 Results per Query", 1, 20, 4, 1,
            help="Number of Colpali search results per rewritten query."
        )
        image_resolution = st.slider(
            "🖼️ Image Resolution (Max Height)", 1024, 3584, 1536, 256,
            help="Maximum height for fetched paper page images (px)."
        )

    # --- Prompt Configuration Expander ---
    with st.expander("📝 Prompt Templates (Jinja2)", expanded=False):
        st.markdown("Use Jinja2 syntax for dynamic prompts.")
        with st.popover("ℹ️ Jinja2 Key Reference"):
            st.markdown(helper_prompt_configuration_jinja2)
            st.markdown("---")
            st.markdown("**Paper Context Keys for Colpali Prompt:**")
            st.code("""
- {{ page }}: Page number
- {{ id }}: Arxiv ID
- {{ doi }}: DOI (nullable)
- {{ date }}: Publication date
- {{ title }}: Paper title
- {{ authors }}: Authors
- {{ abstract }}: Abstract
- {{ url }}: Arxiv URL
- {{ version }}: Paper version
- {{ page_image }}: Page image URL
            """)

        user_prompt_template = st.text_area(
            "🔎 **Search Query Generation Prompt**",
            value=prompt_query_build, height=200,
            help="Template to rewrite user input into search queries. Use {{ prev_chat }} and {{ query_rewrite_count }}."
        )
        system_prompt_template = st.text_area(
            "🤖 **LLM System Prompt**",
            value=prompt_system_llm, height=200,
            help="Overall instructions for the AI assistant's behavior and persona."
        )
        colpali_prompt_template = st.text_area(
            "📑 **Colpali Context Prompt**",
            value=prompt_colpali_content, height=200,
            help="Template for presenting fetched paper content to the LLM. Use paper context keys (see reference)."
        )

    st.divider()
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Colpali Retrieval API Docs](https://llm.arz.ai/docs#/data%20sources/colpali_rag_colpali_arxiv_post)
    - [Embedding Model Info](https://huggingface.co/vidore/colpali-v1.3)
    """)
    st.caption(f"Model ID: `{model_choices[selected_model]}`")


# --- Main Chat Interface ---
st.title("💬 Colpali-Arxiv Chat")
st.markdown(app_description)
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about Arxiv papers, and I'll try to find relevant information using Colpali."}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("What's your question about Arxiv papers?"):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # --- Processing Logic ---
    try:
        # 1. Initialize LLM Client
        client = MagicLLM(
            engine='openai', # Keep as needed
            model=model_choices[selected_model],
            private_key=openai_api_key if openai_api_key else None,
            base_url='https://llm.arz.ai',
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty
        )

        # 2. Prepare Context and Generate Search Queries
        prev_chat_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]]) # Last 6 messages
        chat_query = ModelChat()
        try:
            query_gen_template = Template(user_prompt_template)
            rendered_query_prompt = query_gen_template.render(
                prev_chat=prev_chat_context,
                query_rewrite_count=query_rewrite_count
            )
            chat_query.add_user_message(rendered_query_prompt)
        except TemplateError as e:
            st.error(f"Error rendering search query prompt template: {e}", icon="❌")
            st.stop()

        queries = []
        with st.status("🧠 Generating search queries...", expanded=False) as status:
            try:
                st.write("Calling LLM to generate queries...")
                response = client.llm.generate(chat_query)
                queries_json = response.content.strip().lstrip("```json").rstrip("```")
                queries = json.loads(queries_json)
                if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                     raise ValueError("LLM did not return a valid list of strings for queries.")
                status.update(label=f"✅ Generated {len(queries)} search queries.", state="complete", expanded=False)
            except json.JSONDecodeError as e:
                 st.error(f"Error parsing LLM response for queries (expected JSON list of strings): {e}\nRaw response: `{response.content}`", icon="❌")
                 status.update(label="❌ Error parsing queries", state="error")
                 st.stop()
            except Exception as e:
                 st.error(f"Error generating search queries: {e}", icon="❌")
                 traceback.print_exc() # Log traceback for debugging
                 status.update(label="❌ Error generating queries", state="error")
                 st.stop()

        # Display generated queries in an expander
        with st.expander(f"🔍 Generated Colpali Queries ({len(queries)})", expanded=False):
            for idx, q in enumerate(queries, 1):
                st.markdown(f"- `{q}`")

        # 3. Fetch Colpali Data and Images Concurrently
        all_paper_data = []
        unique_images = set()
        images_data = [] # Store tuples of (item_metadata, encoded_image_data)

        with st.status("🔄 Fetching data from Colpali and processing images...", expanded=True) as status:
            # --- Fetch Colpali paper data ---
            status.update(label="📚 Fetching paper data from Colpali...")
            colpali_results = {} # Store results keyed by query
            try:
                with ThreadPoolExecutor(max_workers=min(5, len(queries))) as executor:
                    future_to_query = {executor.submit(fetch_colpali_data, q, result_count): q for q in queries}
                    completed_count = 0
                    for future in as_completed(future_to_query):
                        query = future_to_query[future]
                        try:
                            papers = future.result()
                            colpali_results[query] = papers
                            processed_papers = 0
                            for paper in papers:
                                # Simple validation - check if essential keys exist
                                if paper.get('page_image') and paper.get('id'):
                                     if paper['page_image'] not in unique_images:
                                         all_paper_data.append(paper)
                                         unique_images.add(paper['page_image'])
                                         processed_papers += 1
                                else:
                                     st.warning(f"Skipping invalid paper data from query '{query}': {paper}", icon="⚠️")
                            completed_count += 1
                            status.update(label=f"📚 Fetched data for query {completed_count}/{len(queries)} ('{query}') - Added {processed_papers} unique items...")
                        except Exception as exc:
                            st.warning(f"Query '{query}' generated an exception during fetch: {exc}", icon="⚠️")
                            colpali_results[query] = [] # Mark as failed but continue
                            completed_count += 1
                            status.update(label=f"⚠️ Error fetching for query {completed_count}/{len(queries)} ('{query}')...")

                if not all_paper_data:
                    st.warning("No valid paper data found from Colpali for the generated queries.", icon="ℹ️")
                    status.update(label="⚠️ No paper data found.", state="complete", expanded=False)
                    st.stop() # Stop if no data to proceed

                status.update(label=f"🖼️ Fetching {len(all_paper_data)} unique images...")

                # --- Fetch and encode images ---
                with ThreadPoolExecutor(max_workers=8) as executor: # More workers for I/O bound image fetching
                    future_to_item = {
                        executor.submit(fetch_and_encode_image, url=item['page_image'], new_height=image_resolution): item
                        for item in all_paper_data
                    }
                    completed_images = 0
                    total_images = len(future_to_item)
                    for future in as_completed(future_to_item):
                        item = future_to_item[future]
                        try:
                            img_encoded = future.result()
                            if img_encoded:
                                images_data.append((item, img_encoded))
                        except Exception as exc:
                            st.warning(f"Failed to fetch or encode image {item.get('page_image', 'N/A')}: {exc}", icon="🖼️")
                        completed_images += 1
                        status.update(label=f"🖼️ Fetched {completed_images}/{total_images} images...")

                status.update(label=f"✅ Fetched {len(images_data)} images successfully.", state="complete", expanded=False)

            except Exception as e:
                 st.error(f"An error occurred during data/image fetching: {e}", icon="❌")
                 traceback.print_exc()
                 status.update(label="❌ Error during data fetching", state="error")
                 st.stop()

        if not images_data:
            st.error("No images could be fetched or processed. Cannot proceed with multimodal generation.", icon="🖼️")
            st.stop()

        # 4. Prepare Final LLM Request with Context and Images
        chat = ModelChat(system=system_prompt_template)
        try:
            colpali_template = Template(colpali_prompt_template)
            for content_dict, image_data in images_data:
                rendered_colpali_prompt = colpali_template.render(**content_dict)
                chat.add_user_message(
                    content=rendered_colpali_prompt,
                    image=image_data,
                    media_type='image/png' # Assuming PNG, adjust if needed
                )
        except TemplateError as e:
            st.error(f"Error rendering Colpali context prompt template: {e}", icon="❌")
            st.stop()
        except Exception as e:
             st.error(f"Error preparing final LLM request: {e}", icon="❌")
             traceback.print_exc()
             st.stop()


        # 5. Generate and Stream Final Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                stream = client.llm.stream_generate(chat)
                for chunk in stream:
                    # Adapt based on actual chunk structure of your client library
                    content_delta = chunk.choices[0].delta.content
                    if content_delta:
                        full_response += content_delta
                        message_placeholder.markdown(full_response + "▌") # Add cursor effect
                message_placeholder.markdown(full_response) # Final response
            except Exception as e:
                 st.error(f"Error during final response generation: {e}", icon="❌")
                 traceback.print_exc()
                 full_response = "Sorry, I encountered an error while generating the response."
                 message_placeholder.markdown(full_response)


        # Add final assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        # Catch-all for unexpected errors in the main flow
        st.error(f"An unexpected error occurred: {e}", icon="🔥")
        traceback.print_exc() # Log for debugging