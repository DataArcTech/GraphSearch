import re
import asyncio
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logging, wrap_embedding_func_with_attrs

from .base import GraphRAGBase

from config import (
    MODEL_NAME,
    LLM_BASE_URL,
    LLM_API_KEY,
    GRAG_MODE
)

class LightRAGMethod(GraphRAGBase):
    def __init__(self, working_dir: str, EMBED_MODEL, top_k: int):
        grag_mode = GRAG_MODE["lightrag"]
        grag = self.init_graphrag(working_dir=working_dir, EMBED_MODEL=EMBED_MODEL)
        super().__init__(grag=grag, QueryParam=QueryParam, grag_mode=grag_mode, top_k=top_k)

    def init_graphrag(self, working_dir:str, EMBED_MODEL):
        
        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
            max_token_size=EMBED_MODEL.max_seq_length,
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return EMBED_MODEL.encode(texts, normalize_embeddings=True)

        async def qwen_complete(
            prompt,
            system_prompt="",
            history_messages=[],
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            return await openai_complete_if_cache(
                model=MODEL_NAME,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                **kwargs,
            )

        grag = LightRAG(
            working_dir=working_dir,
            llm_model_func=qwen_complete,
            embedding_func=embedding_func,
            llm_model_max_async=16,
            max_parallel_insert=16,
            chunk_token_size=400,
            chunk_overlap_token_size=50,
            enable_llm_cache=False,
            log_level=logging.WARNING
        )

        asyncio.run(grag.initialize_storages())
        asyncio.run(initialize_pipeline_status())
        
        return grag

    def context_filter(self, context_data: str, filter_type: str) -> str:
        pattern = (
            r"Knowledge Graph Data \(Entity\):\s*```json\s*(.*?)\s*```.*?"
            r"Knowledge Graph Data \(Relationship\):\s*```json\s*(.*?)\s*```.*?"
            r"Document Chunks.*?```json\s*(.*?)\s*```.*?"
            r"Reference Document List.*?```(?:json|text)?\s*(.*?)\s*```"
        )

        match = re.search(pattern, context_data, re.DOTALL)
        if not match:
            return ""

        entities_str, relations_str, text_chunks_str, reference_list_str = match.groups()

        if filter_type == "semantic":
            return f"""Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
````

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

````json
{reference_list_str}
```"""

        elif filter_type == "relational":
            return f"""Knowledge Graph Data (Entity):

```json
{entities_str}
````

Knowledge Graph Data (Relationship):

````json
{relations_str}
```"""
