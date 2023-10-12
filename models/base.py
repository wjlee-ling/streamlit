from templates import rules, template_en, template_kor

import langchain
from typing import Optional, Any, Union
from langchain.schema import BaseDocumentTransformer
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.document_loaders.base import BaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.cache import InMemoryCache
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class CustomRetrievalChain:
    langchain.llm_cache = InMemoryCache()

    def __init__(
        self,
        prompt: Union[BasePromptTemplate, None] = None,
        llm: Union[BaseLanguageModel, None] = None,
        vectorstore: Union[VectorStore, None] = None,
        condense_question_llm: Optional[BaseLanguageModel] = None,
    ):
        self.prompt = (
            prompt
            if prompt
            else PromptTemplate.from_template(template_kor.replace("{rules}", rules))
        )

        self.llm = (
            ChatOpenAI(
                model_name="gpt-4",
                temperature=0,
            )
            if llm is None
            else llm
        )
        self.vectorstore = (
            Chroma(
                collection_name="default",
            )
            if vectorstore is None
            else vectorstore
        )
        self.retriever = self.vectorstore.as_retriever()
        self.condense_question_llm = (
            ChatOpenAI(
                model_name="gpt-4",
                temperature=0,
            )
            if condense_question_llm is None
            else condense_question_llm
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        # components for loading and indexing raw data
        self.loader = None
        self.splitter = None

        # build a chain with the given components
        self.chain = ConversationalRetrievalChain.from_llm(
            # chain_type:
            # "stuff": default; to use all of the text from the documents in the prompt
            # "map_reduce": to batchify docs and feeds each batch with the question to LLM, and come up with the final answer based on the answers
            # "refine": to batchify docs and feeds the first batch to LLM, and then feeds the second batch with the answer from the first one, and so on
            # "map-rerank": to batchify docs and feeds each batch, return a score and come up with the final answer based on the scores
            self.llm,
            retriever=self.retriever,
            # chain_type_kwargs={"prompt": self.prompt},
            memory=self.memory,
            condense_question_llm=self.condense_question_llm,
            condense_question_prompt=self.prompt,
        )

    def __call__(self):
        return self.chain

    def answer(self, question: str) -> str:
        return self.chain.answer(question)

    @classmethod
    def build_collection(
        cls,
        collection_name: str,
        loader: BaseLoader,
        splitter: BaseDocumentTransformer,
        **kwargs: Any,
    ):
        """Build a new collection for the local Chroma vectorstore"""
        cls.loader = loader
        cls.splitter = splitter

        data = cls.loader.load()
        splits = cls.splitter.split_documents(data)
        cls.vectorstore = Chroma.from_documents(
            collection_name=collection_name,
            documents=splits,
            embedding=OpenAIEmbeddings(),
            collection_metadata={
                "hnsw:space": "cosine"
            },  # default is "l2" (https://docs.trychroma.com/usage-guide)
            **kwargs,
        )


chain = CustomRetrievalChain()
print(chain.llm)
print(chain.vectorstore)
print(chain.retriever)
