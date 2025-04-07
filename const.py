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
You are a helpful assistant tasked with creating a detailed yet concise report. 
Your response must use information provided in paper page images when needed. 
Clearly cite each source within markdown in the format:\n\n
> [Title of Paper](URL), page: X\n\n
Always explicitly reference the Paper ID, title, page number, and URL provided.
""".strip()

prompt_colpali_content = """
Use information from this paper page as necessary. 
When citing this source, reference clearly as:\n
Paper ID: {{id}}, Title: '{{title}}', Page: {{page}}, URL: {{url}}
"""
