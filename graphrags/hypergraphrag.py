import re
import numpy as np

from HyperGraphRAG.hypergraphrag import HyperGraphRAG, QueryParam
from HyperGraphRAG.hypergraphrag.llm import openai_complete_if_cache, GPTKeywordExtractionFormat
from HyperGraphRAG.hypergraphrag.utils import logging, wrap_embedding_func_with_attrs

from .base import GraphRAGBase

from config import (
    MODEL_NAME,
    LLM_BASE_URL,
    LLM_API_KEY,
    GRAG_MODE
)

class HyperGraphRAGMethod(GraphRAGBase):
    def __init__(self, working_dir: str, EMBED_MODEL, top_k: int):
        grag_mode = GRAG_MODE["hypergraphrag"]
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

        grag = HyperGraphRAG(
            working_dir=working_dir,
            llm_model_func=qwen_complete,
            embedding_func=embedding_func,
            chunk_token_size=400,
            chunk_overlap_token_size=50,
            enable_llm_cache=False,
            log_level=logging.WARNING
        )
        
        return grag
    
    def context_filter(self, context_data: str, filter_type: str) -> str:
        pattern = (
            r"-----Entities-----\s+```csv\n(.*?)\n```.*?"
            r"-----Relationships-----\s+```csv\n(.*?)\n```.*?"
            r"-----Sources-----\s+```csv\n(.*?)\n```"
        )
        match = re.search(pattern, context_data, re.DOTALL)

        if not match:
            return ""

        entities, relationships, sources = match.groups()
        
        if filter_type == "semantic":
            return f"""-----Sources-----
```csv
{sources}
````
"""

        elif filter_type == "relational":
            return f"""-----Entities-----
```csv
{entities}
```
-----Relationships-----
```csv
{relationships}
```
"""