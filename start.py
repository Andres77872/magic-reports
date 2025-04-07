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
DEFAULT_MODEL_KEY = list(model_choices.keys())[0]  # Get the first model key as default

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
    # Simple check if default needs key (adjust logic as needed)
    if not openai_api_key and model_choices.get(DEFAULT_MODEL_KEY, "").startswith('@01'):
        st.warning("An API key might be needed for the default or selected model.", icon="⚠️")

    st.divider()  # Visual separator

    st.markdown("## ⚙️ Core Configuration")
    selected_model = st.selectbox(
        "🤖 Select Model",
        list(model_choices.keys()),
        index=0,  # Default to the first model in the list
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
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Hello! Ask me about Arxiv papers, and I'll try to find relevant information using Colpali."}]

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
            engine='openai',  # Keep as needed
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
        prev_chat_context = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])  # Last 6 messages
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
                # Attempt to parse JSON, handling potential markdown code fences
                response_content = response.content.strip()
                if response_content.startswith("```json"):
                    response_content = response_content[len("```json"):].strip()
                if response_content.endswith("```"):
                    response_content = response_content[:-len("```")].strip()

                queries = json.loads(response_content)
                if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                    raise ValueError("LLM did not return a valid list of strings for queries.")
                status.update(label=f"✅ Generated {len(queries)} search queries.", state="complete", expanded=False)
            except json.JSONDecodeError as e:
                st.error(
                    f"Error parsing LLM response for queries (expected JSON list of strings): {e}\nRaw response: `{response.content}`",
                    icon="❌")
                status.update(label="❌ Error parsing queries", state="error")
                st.stop()
            except Exception as e:
                st.error(f"Error generating search queries: {e}", icon="❌")
                traceback.print_exc()  # Log traceback for debugging
                status.update(label="❌ Error generating queries", state="error")
                st.stop()

        # Display generated queries in an expander
        with st.expander(f"🔍 Generated Colpali Queries ({len(queries)})", expanded=False):
            if queries:
                for idx, q in enumerate(queries, 1):
                    st.markdown(f"- `{q}`")
            else:
                st.markdown("No queries were generated.")

        # --- Stages 3 & 4: Fetch Colpali Data and Images Concurrently ---
        all_paper_data = []  # Store validated paper metadata dictionaries
        unique_image_urls = set()  # Track unique image URLs to fetch
        images_data = []  # Store tuples of (item_metadata, encoded_image_data)

        # Use a single status context for the entire fetching process
        with st.status("🔄 Fetching resources...", expanded=True) as status:
            try:
                # --- Stage 3: Fetch Colpali paper data ---
                status.update(label="📚 Fetching paper metadata from Colpali...")
                colpali_results_raw = {}  # Store raw results keyed by query
                papers_processed_count = 0
                total_possible_papers = len(queries) * result_count

                with ThreadPoolExecutor(max_workers=min(5, len(queries) + 1)) as executor:
                    future_to_query = {executor.submit(fetch_colpali_data, q, result_count): q for q in queries}
                    completed_colpali_queries = 0
                    for future in as_completed(future_to_query):
                        query = future_to_query[future]
                        completed_colpali_queries += 1
                        try:
                            papers = future.result()
                            colpali_results_raw[query] = papers
                            valid_papers_in_batch = 0
                            for paper in papers:
                                papers_processed_count += 1
                                # Basic validation
                                if isinstance(paper, dict) and paper.get('page_image') and paper.get('id'):
                                    # Add to list if image URL is new
                                    if paper['page_image'] not in unique_image_urls:
                                        all_paper_data.append(paper)
                                        unique_image_urls.add(paper['page_image'])
                                        valid_papers_in_batch += 1
                                else:
                                    st.warning(
                                        f"Skipping invalid/incomplete paper data from query '{query}': {str(paper)[:100]}...",
                                        icon="⚠️")  # Log snippet

                            status.update(
                                label=f"📚 Fetching metadata... ({completed_colpali_queries}/{len(queries)} queries processed, found {len(all_paper_data)} unique items so far)")

                        except Exception as exc:
                            st.warning(f"Query '{query}' generated an exception during metadata fetch: {exc}",
                                       icon="⚠️")
                            colpali_results_raw[query] = []  # Mark as failed but continue
                            status.update(
                                label=f"⚠️ Error fetching metadata for query {completed_colpali_queries}/{len(queries)} ('{query}')...")

                # Check if any valid paper data was found after trying all queries
                if not all_paper_data:
                    st.warning("No valid paper data with images found from Colpali for the generated queries.",
                               icon="ℹ️")
                    status.update(label="⚠️ No usable paper metadata found.", state="complete", expanded=False)
                    # Optional: Display raw results if debugging needed
                    # with st.expander("Raw Colpali Results (Debug)", expanded=False):
                    #    st.json(colpali_results_raw)
                    st.stop()  # Stop if no data to proceed

                # Metadata fetching complete
                status.update(label=f"✅ Fetched metadata for {len(all_paper_data)} unique paper pages.")
                st.write(
                    f"Found {len(all_paper_data)} unique paper pages with images to process.")  # Give user feedback

                # --- Stage 4: Fetch and encode images ---
                status.update(label=f"🖼️ Fetching and processing {len(all_paper_data)} images...")

                with ThreadPoolExecutor(max_workers=8) as executor:  # More workers for I/O bound image fetching
                    # Map future back to the paper dictionary
                    future_to_item = {
                        executor.submit(fetch_and_encode_image, url=item['page_image'],
                                        new_height=image_resolution): item
                        for item in all_paper_data  # Iterate through the validated list
                    }
                    completed_images = 0
                    total_images = len(future_to_item)
                    for future in as_completed(future_to_item):
                        item_metadata = future_to_item[future]  # Get the associated paper data
                        completed_images += 1
                        try:
                            img_encoded = future.result()  # Get the encoded image data (or None if failed)
                            if img_encoded:
                                images_data.append((item_metadata, img_encoded))  # Pair metadata with its image
                            else:
                                # Handle case where fetch_and_encode_image returned None gracefully
                                st.warning(
                                    f"Image encoding/fetching failed for {item_metadata.get('page_image', 'N/A')}, skipping.",
                                    icon="🖼️")

                        except Exception as exc:
                            st.warning(
                                f"Failed to fetch or encode image {item_metadata.get('page_image', 'N/A')}: {exc}",
                                icon="🖼️")

                        status.update(
                            label=f"🖼️ Processing images... ({completed_images}/{total_images} processed, {len(images_data)} successful)")

                # Image fetching complete
                if not images_data:
                    st.error(
                        "No images could be successfully fetched or processed, even though metadata was found. Cannot proceed.",
                        icon="🖼️")
                    status.update(label="❌ No images processed.", state="error", expanded=True)
                    st.stop()

                status.update(label=f"✅ Fetched and processed {len(images_data)} images.", state="complete",
                              expanded=False)

            except Exception as e:
                st.error(f"An error occurred during data/image fetching: {e}", icon="❌")
                traceback.print_exc()
                status.update(label="❌ Error during resource fetching", state="error", expanded=True)
                st.stop()

        # --- Stage 5: Prepare Final LLM Request ---
        chat = ModelChat(system=system_prompt_template)
        try:
            colpali_template = Template(colpali_prompt_template)
            # Now images_data contains (metadata, image_base64) tuples
            for content_dict, image_base64_data in images_data:
                try:
                    rendered_colpali_prompt = colpali_template.render(**content_dict)
                    chat.add_user_message(
                        content=rendered_colpali_prompt,
                        image=image_base64_data,  # Pass the base64 string
                        media_type='image/png'  # Assuming PNG from fetch_and_encode_image
                    )
                except TemplateError as render_e:
                    st.error(f"Error rendering Colpali context for paper {content_dict.get('id', 'N/A')}: {render_e}",
                             icon="📝")
                    # Decide whether to skip this item or stop
                    st.warning(f"Skipping paper {content_dict.get('id', 'N/A')} due to template error.", icon="⚠️")
                except Exception as add_msg_e:
                    st.error(f"Error adding message for paper {content_dict.get('id', 'N/A')}: {add_msg_e}", icon="❌")
                    st.warning(f"Skipping paper {content_dict.get('id', 'N/A')} due to message error.", icon="⚠️")


        except TemplateError as e:  # Catch error during initial template creation
            st.error(f"Error initializing Colpali context prompt template: {e}", icon="❌")
            st.stop()
        except Exception as e:
            st.error(f"Error preparing final LLM request: {e}", icon="❌")
            traceback.print_exc()
            st.stop()

        # --- Stage 6: Generate and Stream Final Response ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # --- Add Initial Loading/Thinking Indicator ---
                # Display this immediately before starting the potentially blocking stream call
                message_placeholder.markdown("⏳ Thinking...")

                # Ensure the client parameters are set correctly before streaming
                client.llm.temperature = temperature
                client.llm.top_p = top_p
                client.llm.max_tokens = max_tokens
                client.llm.presence_penalty = presence_penalty
                client.llm.frequency_penalty = frequency_penalty
                # Ensure repetition_penalty is handled correctly by your client/model if supported

                # --- Initiate Stream Generation (This call might block until the server starts sending data) ---
                stream = client.llm.stream_generate(chat)

                # --- Process the Stream ---
                first_chunk = True  # Flag to ensure we overwrite the "Thinking..." message cleanly
                for chunk in stream:
                    # Adapt based on actual chunk structure of your client library
                    content_delta = getattr(getattr(getattr(chunk, 'choices', [None])[0], 'delta', None), 'content',
                                            None)

                    if content_delta:
                        if first_chunk:
                            full_response = content_delta  # Start with the first piece of content
                            first_chunk = False
                        else:
                            full_response += content_delta  # Append subsequent content

                        message_placeholder.markdown(
                            full_response + "▌")  # Update placeholder with streaming content + cursor

                # --- Final Cleanup ---
                # Ensure the final full response is displayed without the cursor
                if not full_response:  # Handle cases where the stream might be empty or only contain non-content deltas
                    message_placeholder.markdown("Received an empty response from the model.")
                    full_response = "Received an empty response from the model."  # Set for history
                else:
                    message_placeholder.markdown(full_response)

            except AttributeError as ae:
                error_message = f"Error accessing stream chunk structure: {ae}. Check MagicLLM response format."
                st.error(error_message, icon="🧩")
                traceback.print_exc()
                full_response = "Sorry, I encountered an error processing the response stream."
                message_placeholder.markdown(full_response)  # Update placeholder with error
            except Exception as e:
                error_message = f"Error during final response generation: {e}"
                st.error(error_message, icon="❌")
                traceback.print_exc()
                full_response = "Sorry, I encountered an error while generating the response."
                message_placeholder.markdown(full_response)  # Update placeholder with error

            # Add final assistant response (or error message) to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        # Catch-all for unexpected errors in the main flow outside specific handlers
        st.error(f"An unexpected application error occurred: {e}", icon="🔥")
        traceback.print_exc()  # Log for debugging
        # Optionally add an error message to chat history if appropriate
        if not st.session_state.messages or st.session_state.messages[-1][
            "role"] == "user":  # Avoid double error messages
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Apologies, an unexpected application error occurred: {e}"})
