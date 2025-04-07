prompt_query_build = """
You are assisting in generating relevant search queries for a Retrieval-Augmented Generation (RAG) system based on the user's last message in the conversation. The user's latest message is enclosed within "<user_query>" tags below:

<user_query>
{{prev_chat}}
</user_query>

Your task is to:

1. Analyze the user's query carefully to identify:
   - Main topics and intent
   - Specific details or context implied

2. Generate between 1 to 5 concise and varied search queries ONLY IF ADDITIONAL INFORMATION IS REQUIRED. These queries should:
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
""".strip()

prompt_system_llm = """
You are a helpful assistant tasked with generating a clear, detailed, and concise report based exclusively on information provided in the supplied images of paper pages. Your report must strictly adhere to the following guidelines:

- Always reference and cite the provided source explicitly within your response using the markdown citation format below:

> [Paper ID, Title of Paper](URL), page: X

- Clearly mention each source reference by using its Paper ID, title, specific page number, and the provided URL.
- Carefully integrate resources available from the images, including but not limited to tables, diagrams, and charts.
- When available, replicate tables or diagrams accurately in your response utilizing Mermaid markup syntax.
- NEVER generate or infer information that is not explicitly mentioned or clearly represented in the provided paper page images. Do not include assumptions or external data.

Your resulting report must be precise, evidence-driven, accurately cited, and clearly structured, leveraging all relevant visual and textual sources provided.
""".strip()

prompt_colpali_content = """
Use information from this paper page as necessary. 
When citing this source, reference clearly as:\n
Paper ID: {{id}}, Title: '{{title}}', Page: {{page}}, URL: {{url}}
""".strip()

helper_prompt_configuration_jinja2 = """
Here you will find useful keys available for creating custom prompts in **Jinja2 syntax**.

### 🗨️ Chat Context Keys (`user_prompt_template`):

- `{{ prev_chat }}`:
    - **Type:** String (multiline)
    - **Description:** Recent conversation history between user and assistant. Useful when rephrasing prompts or ensuring continuity.

### 📚 Colpali Paper Data Keys (`colpali_prompt_template`):

Each paper retrieved from Colpali provides the following fields:

- `{{ page }}`:
    - **Type:** Integer
    - **Description:** Page number from the retrieved Arxiv paper.

- `{{ id }}`:
    - **Type:** String
    - **Description:** Unique Arxiv identifier (e.g., "2403.05544").

- `{{ doi }}`:
    - **Type:** String or null
    - **Description:** Digital Object Identifier of the paper, if available.

- `{{ date }}`:
    - **Type:** String
    - **Description:** Formal publication date (e.g., "Mon, 5 Feb 2024 16:12:14 GMT").

- `{{ title }}`:
    - **Type:** String
    - **Description:** Full title of the Arxiv paper.

- `{{ authors }}`:
    - **Type:** String
    - **Description:** List of authors of the paper.

- `{{ abstract }}`:
    - **Type:** String
    - **Description:** Abstract summary provided by the authors.

- `{{ url }}`:
    - **Type:** String (URL)
    - **Description:** Direct URL link to the paper on Arxiv.

- `{{ version }}`:
    - **Type:** String
    - **Description:** Version number of this paper.

- `{{ page_image }}`:
    - **Type:** String (URL)
    - **Description:** Direct URL link to the page image retrieved from Colpali.

### ✨ Example dictionary from a Colpali retrieval result:

```json
{
  "page": 1,
  "id": "2403.05544",
  "doi": null,
  "date": "Mon, 5 Feb 2024 16:12:14 GMT",
  "title": "From Algorithm Worship to the Art of Human Learning: Insights from 50-year journey of AI in Education",
  "authors": "Kaska Porayska-Pomsta",
  "abstract": "Current discourse surrounding Artificial Intelligence (AI)...",
  "url": "https://arxiv.org/abs/2403.05544",
  "version": "1",
  "page_image": "https://llm.arz.ai/rag/colpali/arxiv/2403.05544v1_p_1.png"
}
```

### 💡 Usage examples in your templates:

```jinja
Please provide a summary of the following paper titled "{{ title }}", authored by {{ authors }}. 

Abstract:
{{ abstract }}

Published on: {{ date }}
Link to paper: [{{ url }}]({{ url }})
```

These keys help you to dynamically render templates and present paper information clearly to the AI assistant, enhancing context-awareness and response quality.
""".strip()
