from pathlib import Path
from typing import Any, Dict

from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from config import DocumentProcessingConfig


class QueryEngine:
    def __init__(self, document_store, config: DocumentProcessingConfig):
        self.config = config
        self.document_store = document_store
        self._setup_pipeline()

    def _setup_pipeline(self):
        prompt_template = """
        You are an expert in the subject matter.
        You provide answers based on the following context.
        Instructions:
        - Answer the question truthfully using the information provided.
        - If multiple documents contain relevant information, combine them to form a comprehensive answer.
        - Do not use information outside of the provided sources.
        - If no relevant information is found, state that directly.
        - Given these documents, answer the question.\nDocuments:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            \nQuestion: {{query}}
            \nAnswer:
        """

        components = [
            (
                "text_embedder",
                SentenceTransformersTextEmbedder(
                    model="sentence-transformers/all-MiniLM-L6-v2"
                ),
            ),
            (
                "retriever",
                QdrantEmbeddingRetriever(document_store=self.document_store),
            ),
            ("ranker", TransformersSimilarityRanker(model=self.config.ranker_model)),
            ("prompt_builder", PromptBuilder(template=prompt_template)),
            ("llm", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"))),
            ("answer_builder", AnswerBuilder()),
        ]

        self.pipeline = Pipeline()
        for name, component in components:
            self.pipeline.add_component(instance=component, name=name)

        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "ranker.documents")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")
        self.pipeline.connect("llm.replies", "answer_builder.replies")
        self.pipeline.connect("llm.meta", "answer_builder.meta")
        self.pipeline.connect("retriever", "answer_builder.documents")

    def query(self, query: str) -> Dict[str, Any]:
        result = self.pipeline.run(
            {
                "text_embedder": {"text": query},
                "ranker": {"query": query, "top_k": self.config.top_k},
                "prompt_builder": {"query": query},
                "answer_builder": {"query": query},
            }
        )

        answer = result["answer_builder"]["answers"][0].data
        contexts = result["ranker"]["documents"]

        top_context = contexts[0] if contexts else None

        source_info = None
        if top_context:
            source_info = {
                "source": Path(top_context.meta["file_path"]).name,
                "page": top_context.meta.get("page_number", 1),
                "score": top_context.score if hasattr(top_context, "score") else 0.0,
                "content": top_context.content,
            }

        return {"answer": answer, "source": source_info}
