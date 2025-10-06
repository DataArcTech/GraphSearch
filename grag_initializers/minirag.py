import numpy as np
from transformers import AutoTokenizer

from minirag import MiniRAG
from minirag.llm.openai import openai_complete_if_cache, GPTKeywordExtractionFormat
from minirag.utils import logging, wrap_embedding_func_with_attrs

from config import (
    MODEL_NAME,
    LLM_BASE_URL,
    LLM_API_KEY,
)

def init_minirag(working_dir:str, EMBED_MODEL):

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
    
    grag = MiniRAG(
        working_dir=working_dir,
        llm_model_func=qwen_complete,
        embedding_func=embedding_func,
        max_parallel_insert=16,
        chunk_token_size=400,
        chunk_overlap_token_size=50,
        log_level=logging.WARNING
    )