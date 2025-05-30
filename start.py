import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from jinja2 import Template, TemplateError
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

from const import (
    prompt_query_build,
    prompt_system_llm,
    prompt_colpali_content,
    helper_prompt_configuration_jinja2,
    model_choices,
    app_description
)
from utils import fetch_and_encode_image, fetch_colpali_data

st.set_page_config(
    page_title="Colpali-Arxiv AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://llm.arz.ai/docs',
        'Report a bug': "mailto:andres@arz.ai",
        'About': f"## Colpali-Arxiv AI Chatbot\n{app_description}"
    }
)

DEFAULT_MODEL_KEY = list(model_choices.keys())[0]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_used" not in st.session_state:
    st.session_state.sources_used = []
if "last_usage" not in st.session_state:
    st.session_state.last_usage = None

if "new_user_input" not in st.session_state:
    st.session_state.new_user_input = False

if "regen_trigger" not in st.session_state:
    st.session_state.regen_trigger = False

with st.sidebar:
    st.markdown("# Colpali Chat")
    st.caption("AI Assistant for Arxiv")

    st.divider()

    st.markdown("### 🔑 API Credentials")
    openai_api_key = st.text_input(
        "Magic-LLM API Key (optional)",
        type="password",
        placeholder="Enter your key if required",
        help="Needed for certain models (e.g., starting with '@01')."
    )
    if not openai_api_key and model_choices.get(DEFAULT_MODEL_KEY, "").startswith('@01'):
        st.info("An API key might be needed for the default or selected model.", icon="ℹ️")

    st.divider()

    st.markdown("### ⚙️ Core Configuration")
    selected_model = st.selectbox(
        "🤖 Select Model",
        list(model_choices.keys()),
        index=0,
        help="Choose the Large Language Model for generating responses."
    )
    st.caption(f"Model ID: `{model_choices[selected_model]}`")

    with st.expander("🛠️ Advanced Settings"):
        st.markdown("##### LLM Generation Parameters")
        temperature = st.slider(
            "🌡️ Temperature", 0.0, 2.0, 0.75, 0.01,
            help="Controls randomness (0=deterministic, 2=very random)."
        )
        top_p = st.slider(
            "🎯 Top-P", 0.0, 1.0, 0.95, 0.01,
            help="Nucleus sampling threshold (1=consider all)."
        )
        max_tokens = st.number_input(
            "📝 Max New Tokens", 1, 8192, 4096, 128,
            help="Max length of generated response."
        )
        col_pen1, col_pen2, col_pen3 = st.columns(3)
        with col_pen1:
            presence_penalty = st.slider(
                "👤 Presence", -2.0, 2.0, 0.0, 0.01,
                help="Penalty for new token presence."
            )
        with col_pen2:
            frequency_penalty = st.slider(
                "🔄 Frequency", -2.0, 2.0, 0.0, 0.01,
                help="Penalty for new token frequency."
            )
        with col_pen3:
            repetition_penalty = st.slider(
                "♻️ Repetition", 0.5, 2.0, 1.0, 0.01,
                help="Penalty for overall repetition (>1 less likely)."
            )

        st.markdown("##### Colpali Search Parameters")
        col_srch1, col_srch2, col_srch3 = st.columns(3)
        with col_srch1:
            query_rewrite_count = st.number_input(
                "✍️ Rewrites", 1, 10, 5, 1,
                help="Number of search queries generated from your input."
            )
        with col_srch2:
            result_count = st.number_input(
                "📚 Results/Q", 1, 20, 4, 1,
                help="Number of Colpali search results per query."
            )
        with col_srch3:
            image_resolution = st.select_slider(
                "🖼️ Image Res (H)",
                options=[512, 768, 1024, 1536, 2048, 3072, 3584],
                value=1536,
                help="Max height for fetched images (px). Larger images use more tokens."
            )

    with st.expander("📝 Prompt Templates (Jinja2)"):
        st.markdown("Use Jinja2 syntax. See reference below.")
        with st.popover("ℹ️ Jinja2 Key Reference"):
            st.markdown(helper_prompt_configuration_jinja2)

        user_prompt_template = st.text_area(
            "🔎 **Search Query Generation Prompt**",
            value=prompt_query_build, height=150,
            help="Template to rewrite user input into search queries. Vars: {{ prev_chat }}, {{ query_rewrite_count }}."
        )
        system_prompt_template = st.text_area(
            "🤖 **LLM System Prompt**",
            value=prompt_system_llm, height=150,
            help="Overall instructions for the AI assistant's behavior."
        )
        colpali_prompt_template = st.text_area(
            "📑 **Colpali Context Prompt**",
            value=prompt_colpali_content, height=150,
            help="Template for presenting fetched paper content. Use paper context keys (see reference)."
        )

    st.divider()
    st.markdown("##### 📚 Resources")
    st.markdown("""
    - [Colpali Retrieval API Docs](https://llm.arz.ai/docs#/data%20sources/colpali_rag_colpali_arxiv_post)
    - [Embedding Model Info](https://huggingface.co/vidore/colpali-v1.3)
    """)

st.title("💬 Colpali-Arxiv Chat")
st.markdown(app_description)
st.divider()

for idx, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]

    # Display chat bubbles
    st.chat_message(role).markdown(content)

    # Only show sources after the last assistant message
    is_last_assistant = (
            role == "assistant" and idx == len(st.session_state.messages) - 1
    )

    row1, row2 = st.columns([1, 1])

    with row1:
        if is_last_assistant and st.session_state.sources_used:
            with st.expander("### 📚 **Sources Used**", expanded=False):
                for src in st.session_state.sources_used:
                    title = src.get("title", "Untitled")
                    page = src.get("page", "N/A")
                    arxiv_id = src.get("id")
                    page_image = src.get("page_image", None)

                    # Compose display text with title, page, and Arxiv link
                    info_text = f"**{title}**  \n_Page: {page}_"
                    if arxiv_id:
                        info_text += f"  \n[ArXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"

                    # Layout: Text with optional thumbnail
                    if page_image and page_image.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        col_text, col_img = st.columns([5, 1])
                        with col_text:
                            st.markdown(info_text, unsafe_allow_html=True)
                        with col_img:
                            st.image(page_image, width=80, caption="", use_container_width=False)
                    else:
                        st.markdown(info_text, unsafe_allow_html=True)

                st.caption("🔗 These sources were retrieved from the Colpali database and used as context.")
    with row2:
        if is_last_assistant and st.session_state.last_usage is not None:
            usage = st.session_state.last_usage.usage
            with st.expander("📊 **Token Usage Details**", expanded=False):
                st.caption("Breakdown of tokens consumed and timing stats during LLM generation:")
                rows = [
                    ("🔢 Prompt tokens", usage.prompt_tokens),
                    ("📝 Completion tokens", usage.completion_tokens),
                    ("🧮 Total tokens", usage.total_tokens),
                    ("⏱️ Time To First Token (s)", f"{usage.ttft:.2f}"),
                    ("🏁 Total Time Finish (s)", f"{usage.ttf:.2f}"),
                    ("⚡ Tokens/sec", f"{usage.tps:.2f}"),
                ]
                for label, value in rows:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**{label}**")
                    with col2:
                        st.code(str(value))

regen_button_holder = st.empty()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if regen_button_holder.button("🔄 Regenerate response"):
        st.session_state.regen_trigger = True
        last_user_idx = None
        for idx in range(len(st.session_state.messages) - 2, -1, -1):
            if st.session_state.messages[idx]["role"] == "user":
                last_user_idx = idx
                break
        if last_user_idx is not None:
            st.session_state.messages = st.session_state.messages[:last_user_idx + 1]
            st.session_state.new_user_input = True
            st.rerun()

if prompt := st.chat_input("What's your question about Arxiv papers?"):
    st.session_state.new_user_input = True
    st.session_state.sources_used = []
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt:
    st.chat_message("user").markdown(prompt)

if (
        (st.session_state.get("new_user_input", False))
        and
        (st.session_state.messages and st.session_state.messages[-1]["role"] == "user")
):

    prompt = st.session_state.messages[-1]["content"]

    try:
        # --- Stage 1: Initialize LLM Client ---
        client = MagicLLM(
            engine='openai',
            model=model_choices[selected_model],
            private_key=openai_api_key if openai_api_key else None,
            base_url='https://llm.arz.ai',
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )

        thinking_placeholder = st.chat_message("assistant").empty()
        thinking_placeholder.markdown("🤔 Thinking...")

        # --- Stage 2: Prepare Context and Generate Search Queries ---
        prev_chat_context = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-8:]]
        )

        chat_query = ModelChat()
        queries = []
        query_gen_error = None
        try:
            query_gen_template = Template(user_prompt_template)
            rendered_query_prompt = query_gen_template.render(
                prev_chat=prev_chat_context,
                query_rewrite_count=query_rewrite_count
            )
            chat_query.add_user_message(rendered_query_prompt)

            with st.status("🧠 Generating search queries...", expanded=False) as status:
                st.write(f"Asking {selected_model} to create {query_rewrite_count} search terms...")
                response = client.llm.generate(chat_query)
                response_content = response.content.strip()

                if response_content.startswith("```json"):
                    response_content = response_content[len("```json"):].strip()
                if response_content.endswith("```"):
                    response_content = response_content[:-len("```")].strip()

                try:
                    queries = json.loads(response_content)
                    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                        raise ValueError("LLM did not return a valid list of strings for queries.")
                    queries = queries[:query_rewrite_count]
                    status.update(label=f"✅ Generated {len(queries)} queries.", state="complete", expanded=False)
                except (json.JSONDecodeError, ValueError) as e:
                    query_gen_error = f"Error parsing LLM query response (expected JSON list): {e}\nRaw: `{response.content}`"
                    status.update(label="⚠️ Error parsing queries", state="error")

        except TemplateError as e:
            query_gen_error = f"Error rendering search query template: {e}"
        except Exception as e:
            query_gen_error = f"Error generating search queries: {e}"
            traceback.print_exc()

        if query_gen_error:
            st.error(query_gen_error, icon="❌")
            thinking_placeholder.markdown(f"Sorry, I encountered an error generating search queries: {query_gen_error}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {query_gen_error}"})
            st.stop()

        if not queries:
            st.warning("The LLM didn't generate any search queries based on your input.", icon="🤷")
            thinking_placeholder.markdown(
                "I couldn't determine relevant search terms from your request. Could you please rephrase?"
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": "I couldn't determine relevant search terms. Please rephrase."}
            )
            st.stop()

        with st.expander(f"🔍 Using {len(queries)} Colpali Queries", expanded=False):
            st.markdown("\n".join([f"- `{q}`" for q in queries]))

        # --- Stages 3 & 4: Fetch Colpali Data and Images Concurrently ---
        all_paper_data = []
        unique_image_urls = set()
        images_data = []
        fetch_errors = []

        with st.status("⏳ Fetching resources...", expanded=True) as status:
            try:
                # --- Stage 3: Fetch Colpali paper data ---
                status.update(label=f"📚 Searching Colpali with {len(queries)} queries...")
                colpali_results_raw = {}
                papers_processed_count = 0
                max_papers_to_consider = len(queries) * result_count

                with ThreadPoolExecutor(max_workers=min(8, len(queries) + 1)) as executor:
                    future_to_query = {executor.submit(fetch_colpali_data, q, result_count): q for q in queries}
                    completed_colpali_queries = 0

                    for future in as_completed(future_to_query):
                        query = future_to_query[future]
                        completed_colpali_queries += 1
                        progress = completed_colpali_queries / len(queries)
                        status.update(
                            label=f"📚 Fetching metadata... Query {completed_colpali_queries}/{len(queries)} ('{query[:30]}...')"
                        )

                        try:
                            papers = future.result()
                            colpali_results_raw[query] = papers
                            valid_papers_in_batch = 0
                            for paper in papers:
                                papers_processed_count += 1
                                if isinstance(paper, dict) and paper.get('page_image') and paper.get('id'):
                                    img_url = paper['page_image']
                                    if img_url not in unique_image_urls:
                                        all_paper_data.append(paper)
                                        unique_image_urls.add(img_url)

                        except Exception as exc:
                            warning_msg = f"Query '{query}' failed during metadata fetch: {exc}"
                            st.warning(warning_msg, icon="⚠️")
                            fetch_errors.append(warning_msg)
                            colpali_results_raw[query] = {"error": str(exc)}

                if not all_paper_data:
                    st.warning("No relevant paper pages found in Colpali for the generated queries.", icon="ℹ️")
                    status.update(label="🤷 No relevant papers found.", state="complete", expanded=False)
                    thinking_placeholder.markdown("I couldn't find relevant information in Colpali for your query.")
                    st.session_state.messages.append({"role": "assistant", "content": "Couldn't find relevant papers."})
                    st.stop()

                status.update(label=f"✅ Found {len(all_paper_data)} unique paper pages. Fetching images...")
                st.write(f"Found {len(all_paper_data)} unique items.")

                # --- Stage 4: Fetch and encode images ---
                images_to_fetch = len(all_paper_data)
                status.update(label=f"🖼️ Processing {images_to_fetch} images...")

                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_item = {
                        executor.submit(
                            fetch_and_encode_image,
                            url=item['page_image'],
                            new_height=image_resolution
                        ): item for item in all_paper_data
                    }
                    completed_images = 0
                    for future in as_completed(future_to_item):
                        item_metadata = future_to_item[future]
                        completed_images += 1
                        progress = completed_images / images_to_fetch
                        status.update(label=f"🖼️ Processing images... ({completed_images}/{images_to_fetch})")
                        try:
                            img_encoded = future.result()
                            if img_encoded:
                                images_data.append((item_metadata, img_encoded))
                            else:
                                warning_msg = f"Image fetch/encode failed for {item_metadata.get('page_image', 'N/A')}, skipping."
                                st.caption(warning_msg)
                                fetch_errors.append(warning_msg)

                        except Exception as exc:
                            warning_msg = f"Failed image {item_metadata.get('page_image', 'N/A')}: {exc}"
                            st.caption(warning_msg)
                            fetch_errors.append(warning_msg)

                if not images_data:
                    st.error(
                        "Could not fetch or process any images for the found papers. Cannot generate response.",
                        icon="🖼️"
                    )
                    status.update(label="❌ No images processed.", state="error", expanded=True)
                    thinking_placeholder.markdown(
                        "Sorry, I found papers but couldn't load their images to understand the content."
                    )
                    st.session_state.messages.append({"role": "assistant", "content": "Error loading paper images."})
                    st.stop()

                st.session_state.sources_used = [item_data for item_data, _ in images_data]

                status.update(
                    label=f"✅ Processed {len(images_data)} images. Preparing final answer...",
                    state="complete", expanded=False
                )
                time.sleep(0.5)

            except Exception as e:
                st.error(f"An error occurred during data/image fetching: {e}", icon="❌")
                traceback.print_exc()
                status.update(label="❌ Error fetching resources", state="error", expanded=True)
                thinking_placeholder.markdown(f"Sorry, an error occurred while fetching resources: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error fetching resources: {e}"})
                st.stop()

        # --- Stage 5: Prepare Final LLM Request ---
        thinking_placeholder.markdown("📝 Preparing final response using retrieved context...")
        final_chat = ModelChat()

        try:
            system_template = Template(system_prompt_template)
            rendered_system_prompt = system_template.render()
            final_chat.set_system(rendered_system_prompt)
        except TemplateError as e:
            st.warning(f"Error rendering system prompt template (using default behavior): {e}", icon="⚠️")
        except Exception as e:
            st.warning(f"Unexpected error setting system prompt: {e}", icon="⚠️")

        history_limit = 4
        relevant_history = st.session_state.messages[-(2 * history_limit):-1]
        for msg in relevant_history:
            if msg["role"] == "user":
                final_chat.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                final_chat.add_assistant_message(msg["content"])

        context_added_count = 0
        try:
            colpali_template = Template(colpali_prompt_template)
            for content_dict, image_base64_data in images_data:
                try:
                    rendered_colpali_prompt = colpali_template.render(**content_dict)
                    final_chat.add_user_message(
                        content=rendered_colpali_prompt,
                        image=image_base64_data,
                        media_type='image/png'
                    )
                    context_added_count += 1
                except TemplateError as render_e:
                    st.warning(
                        f"Skipping paper {content_dict.get('id', 'N/A')} due to template render error: {render_e}",
                        icon="📝"
                    )
                except Exception as add_msg_e:
                    st.warning(
                        f"Skipping paper {content_dict.get('id', 'N/A')} due to message adding error: {add_msg_e}",
                        icon="🧩"
                    )

            if context_added_count == 0 and len(images_data) > 0:
                st.error("Failed to prepare context from any of the fetched papers.", icon="❌")
                thinking_placeholder.markdown(
                    "Sorry, I encountered an error preparing the context from the fetched papers.")
                st.session_state.messages.append({"role": "assistant", "content": "Error preparing context."})
                st.stop()

        except TemplateError as e:
            st.error(f"Fatal Error: Cannot initialize Colpali context template: {e}", icon="❌")
            thinking_placeholder.markdown(f"Sorry, a critical error occurred with the prompt template: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Prompt template error: {e}"})
            st.stop()
        except Exception as e:
            st.error(f"Error preparing final LLM request context: {e}", icon="❌")
            traceback.print_exc()
            thinking_placeholder.markdown(f"Sorry, an error occurred preparing the final request: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error preparing request: {e}"})
            st.stop()

        final_chat.add_user_message(prompt)

        # --- Stage 6: Generate and Stream Final Response ---
        message_placeholder = thinking_placeholder
        message_placeholder.markdown("⏳ Generating final answer...")

        full_response = ""
        try:
            stream = client.llm.stream_generate(final_chat)

            first_chunk = True
            for chunk in stream:
                if chunk is not None:
                    st.session_state.last_usage = chunk
                content_delta = None
                try:
                    if chunk and chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta:
                            content_delta = delta.content
                except (AttributeError, IndexError):
                    pass

                if content_delta:
                    if first_chunk:
                        full_response = content_delta
                        first_chunk = False
                    else:
                        full_response += content_delta
                    message_placeholder.markdown(full_response + "▌")

            if not full_response and not first_chunk:
                message_placeholder.markdown("Received an empty response from the model.")
                full_response = "Received an empty response from the model."
            elif first_chunk:
                message_placeholder.markdown("No response generated by the model.")
                full_response = "No response generated."
            else:
                message_placeholder.markdown(full_response)

        except AttributeError as ae:
            error_message = f"Error processing stream: {ae}. Check MagicLLM response format."
            st.error(error_message, icon="🧩")
            traceback.print_exc()
            full_response = "Sorry, I encountered an error processing the response stream."
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_message = f"Error during final response generation: {e}"
            st.error(error_message, icon="❌")
            traceback.print_exc()
            full_response = f"Sorry, I encountered an error: {e}"
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        st.session_state.new_user_input = False
        st.session_state.regen_trigger = False
        time.sleep(0.1)
        st.rerun()

    except Exception as e:
        st.error(f"An unexpected application error occurred: {e}", icon="🔥")
        traceback.print_exc()
        try:
            thinking_placeholder.markdown(f"An unexpected error occurred: {e}")
        except NameError:
            st.chat_message("assistant").error(f"An unexpected error occurred: {e}")

        if not st.session_state.messages or st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Apologies, an unexpected error occurred: {e}"}
            )
        st.session_state.new_user_input = False
        st.session_state.regen_trigger = False

if "new_user_input" in st.session_state and not st.session_state.new_user_input:
    pass
elif "new_user_input" in st.session_state:
    st.session_state.new_user_input = False
