import json
import re
import os
import asyncio
from typing import Any, Dict, List, Tuple
from openai import AsyncOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, MODEL_NAME, EMBED_MODEL_NAME

def compute_args_hash(*args: Any, cache_type: str | None = None) -> str:
    """Compute a hash for the given arguments.
    Args:
        *args: Arguments to hash
        cache_type: Type of cache (e.g., 'keywords', 'query', 'extract')
    Returns:
        str: Hash string
    """
    import hashlib

    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])
    if cache_type:
        args_str = f"{cache_type}:{args_str}"

    # Compute MD5 hash
    return hashlib.md5(args_str.encode()).hexdigest()

async def openai_complete(
    prompt,
    model="gpt-4o",
    system_prompt=None,
    history_messages=[],
    temperature=0.0,
    **kwargs,
) -> str:
    model = MODEL_NAME
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )

    return response.choices[0].message.content

def load_vdb(dataset, documents):
    import faiss
    from sentence_transformers import SentenceTransformer
    index_path = f"./db/vdb/{dataset}_index.faiss"
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index, embed_model
    
    embeddings = embed_model.encode(documents, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    return index, embed_model

def vdb_retrieve(question, documents, index, embed_model, top_k):
    query_embedding = embed_model.encode([question])
    _, I = index.search(query_embedding, k=top_k)
    retrieved_docs = [documents[i] for i in I[0]]
    return retrieved_docs

# def context_filter(context_data: str, filter_type: str) -> str:
#     def clean_json_block(json_str: str) -> str:
#         data = json.loads(json_str)
#         for item in data:
#             item.pop("created_at", None)
#             item.pop("file_path", None)
#             item.pop("id", None)
#             item.pop("rank", None)
#         return json.dumps(data, separators=(',', ':'), ensure_ascii=False)

#     pattern = r"-----Entities\(KG\)-----\s+```json\n(.*?)\n```.*?-----Relationships\(KG\)-----\s+```json\n(.*?)\n```.*?-----Document Chunks\(DC\)-----\s+```json\n(.*?)\n```"
#     match = re.search(pattern, context_data, re.DOTALL)

#     if not match:
#         return context_data

#     entities_str, relations_str, text_units_str = match.groups()

#     cleaned_entities = clean_json_block(entities_str)
#     cleaned_relations = clean_json_block(relations_str)
#     cleaned_text_units = clean_json_block(text_units_str)
    
#     return f"""-----Entities(KG)-----

# ```json
# {cleaned_entities}
# ```

# -----Relationships(KG)-----

# ```json
# {cleaned_relations}
# ```

# -----Document Chunks(DC)-----

# ```json
# {cleaned_text_units}
# ```

# """

def context_filter(grag_name:str, context_data: str, filter_type: str) -> str:
    if grag_name == "lightrag":
        pattern = r"-----Entities\(KG\)-----\s+```json\n(.*?)\n```.*?-----Relationships\(KG\)-----\s+```json\n(.*?)\n```.*?-----Document Chunks\(DC\)-----\s+```json\n(.*?)\n```"
        match = re.search(pattern, context_data, re.DOTALL)

        if not match:
            return ""

        entities, relationships, chunks = match.groups()
        
        if filter_type == "text":
            return f"""-----Document Chunks(DC)-----

```json
{chunks}
```

"""

        elif filter_type == "graph":
            return f"""-----Entities(KG)-----

```json
{entities}
```

-----Relationships(KG)-----

```json
{relationships}
```

"""
    
    elif grag_name == "hypergraphrag":
        pattern = (
            r"-----Entities-----\s+```csv\n(.*?)\n```.*?"
            r"-----Relationships-----\s+```csv\n(.*?)\n```.*?"
            r"-----Sources-----\s+```csv\n(.*?)\n```"
        )
        match = re.search(pattern, context_data, re.DOTALL)

        if not match:
            return ""

        entities, relationships, sources = match.groups()
        
        if filter_type == "text":
            return f"""-----Sources-----
```csv
{sources}
````
"""

        elif filter_type == "graph":
            return f"""-----Entities-----
```csv
{entities}
```
-----Relationships-----
```csv
{relationships}
```
"""

    elif grag_name == "pathrag":
        pattern = (
            r"-----global-information-----.*?"
            r"-----high-level entity information-----\s+```csv\n(.*?)\n```.*?"
            r"-----high-level relationship information-----\s+```csv\n(.*?)\n```.*?"
            r"-----Sources-----\s+```csv\n(.*?)\n```.*?"
            r"-----local-information-----.*?"
            r"-----low-level entity information-----\s+```csv\n(.*?)\n```.*?"
            r"-----low-level relationship information-----\s+```csv\n(.*?)\n```"
        )
        match = re.search(pattern, context_data, re.DOTALL)

        try:
            hl_entities, hl_relations, sources, ll_entities, ll_relations = match.groups()
        except:
            return ""

        if filter_type == "text":
            return f"""-----Sources-----
```csv
{sources}
````
"""

        elif filter_type == "graph":
            return f"""-----global-information-----
-----high-level entity information-----
```csv
{hl_entities}
```
-----high-level relationship information-----
```csv
{hl_relations}
```
-----local-information-----
-----low-level entity information-----
```csv
{ll_entities}
```
-----low-level relationship information-----
```csv
{ll_relations}
```
"""

def extract_words_str(text):
    return ' '.join(re.findall(r'[A-Za-z]+', text))

def format_history_context(history):
    history_context_str = ""
    for i, (q, ctx_sum, a) in enumerate(history):
        history_context_str += f"Sub-query {i + 1}: {q}\nRetrieved context:\n{ctx_sum}\nSub-query answer: {a}\n\n"
    return history_context_str.strip()

def truncate_str_by_token_size(
    text: str,
    max_token_size: int,
    tokenizer
) -> str:
    if max_token_size <= 0:
        return ""

    encoded = tokenizer.encode(text)
    truncated = encoded[:max_token_size]
    return tokenizer.decode(truncated, skip_special_tokens=True)

def normalize(text: str) -> str:
    import string
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        text (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(text)))).split()

def parse_expanded_queries(query_expansion_result: str):
    import ast
    """
    Try to extract and parse a Python-style list of strings from
    the model output, even if surrounded by extra text.
    """
    text = query_expansion_result.strip()

    # Step 1️⃣: 尝试直接安全解析
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    # Step 2️⃣: 用正则在文本中提取形如 ["...", "..."] 的部分
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        list_str = match.group(0)
        try:
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    # Step 3️⃣: 如果都失败，就退化为单元素列表
    return [text]
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        # logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop
