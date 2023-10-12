from .templates import rules, template_en, template_kor
from utils import create_collection

import langchain
from typing import Optional, Any, Dict
from langchain.schema import BaseDocumentTransformer
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.cache import InMemoryCache
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class BaseBot:
    langchain.llm_cache = InMemoryCache()

    def __init__(
        self,
        prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[BaseLanguageModel] = None,
        vectorstore: Optional[VectorStore] = None,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        configs: Optional[Dict[str, Any]] = None,
    ) -> None:
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

    def __call__(self, question: str):
        return self.chain(question)

    @staticmethod
    def __configure__(configs: Dict[str, Any]):
        """
        각 컴포넌트에 kwargs로 들어가는 인자들의 값을 설정합니다. 사용자가 설정하지 않은 값들의 기본값을 설정합니다.

        TO-DO:
        - choose size appropriate to llm context size
        """
        default_configs = {}

        splitter_configs = (
            configs.get(
                "splitter", {"chunk_size": 500, "chunk_overlap": 20}
            )  # default: 4000 / 200 # TO-DO
            if configs
            else {"chunk_size": 500, "chunk_overlap": 20}
        )
        default_configs["splitter"] = splitter_configs
        return default_configs

    @classmethod
    def from_new_collection(
        cls,
        loader: BaseLoader,
        splitter: Optional[BaseDocumentTransformer] = None,
        collection_name: str = "default",
        prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[BaseLanguageModel] = None,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        configs: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """Build new collection AND chain based on it"""
        configs = cls.__configure__(configs)
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            **configs["splitter"],
        )

        docs = splitter.split_documents(data)
        vectorstore = create_collection(
            collection_name=collection_name,
            docs=docs,
        )
        return cls(
            prompt=prompt,
            llm=llm,
            vectorstore=vectorstore,
            condense_question_llm=condense_question_llm,
            configs=configs,
        )
