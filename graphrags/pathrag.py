import re
import numpy as np

from PathRAG.PathRAG import PathRAG, QueryParam
from PathRAG.PathRAG.llm import openai_complete_if_cache, GPTKeywordExtractionFormat
from PathRAG.PathRAG.utils import logging, wrap_embedding_func_with_attrs

from config import (
    MODEL_NAME,
    LLM_BASE_URL,
    LLM_API_KEY,
    GRAG_MODE
)

from .base import GraphRAGBase

class PathRAGMethod(GraphRAGBase):
    def __init__(self, working_dir: str, EMBED_MODEL, top_k: int):
        grag_mode = GRAG_MODE["pathrag"]
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

        grag = PathRAG(
            working_dir=working_dir,
            llm_model_func=qwen_complete,
            embedding_func=embedding_func,
            chunk_token_size=400,
            chunk_overlap_token_size=50,
            # enable_llm_cache=False,
            log_level=logging.WARNING
        )
        
        return grag
    
    def context_filter(self, context_data: str, filter_type: str) -> str:
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

        if filter_type == "semantic":
            return f"""-----Sources-----
```csv
{sources}
````
"""

        elif filter_type == "relational":
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