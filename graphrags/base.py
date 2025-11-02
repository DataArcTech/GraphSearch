from abc import ABC, abstractmethod

class GraphRAGBase(ABC):
    def __init__(self, grag, QueryParam, grag_mode, top_k):
        self.grag = grag
        self.QueryParam = QueryParam
        self.grag_mode = grag_mode
        self.top_k = top_k

    @abstractmethod
    def init_graphrag(self, working_dir: str, EMBED_MODEL):
        pass

    @abstractmethod
    def context_filter(self, context_data: str, filter_type: str) -> str:
        pass

    async def aquery_context(self, question: str):
        return await self.grag.aquery(
            question,
            self.QueryParam(mode=self.grag_mode, only_need_context=True, top_k=self.top_k)
        )
    
    async def aquery_answer(self, question: str):
        return await self.grag.aquery(
            question,
            self.QueryParam(mode=self.grag_mode, only_need_context=False, top_k=self.top_k)
        )