from .templates import CONDENSE_QUESTION_TEMPLATE, STUFF_QA_TEMPLATE
from utils import create_collection

import langchain
from typing import Optional, Any, Dict, Union
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
from pydantic import BaseModel


# class CustomPrompts(BaseModel):
#     """
#     Prompts for each chain type: 'stuff', 'map_reduce', 'refine', 'map-rerank'
#     Refer to [langchain.chains.question_answering](https://github.com/langchain-ai/langchain/tree/c2d1d903fa35b91018b4d777db2b008fcbaa9fbc/langchain/chains/question_answering) for default prompts.
#     """

#     condense_question_prompt: BasePromptTemplate  # for first question condesing w/ context
#     qa_prompt: BasePromptTemplate  # for final answer generation
#     combine_prompt: Optional[BasePromptTemplate] = None  # for "map_reduce"
#     collapse_prompt: Optional[BasePromptTemplate] = None  # for "map_reduce"
#     refine_prompt: Optional[BasePromptTemplate] = None  # for "refine"


class BaseBot:
    langchain.llm_cache = InMemoryCache()

    def __init__(
        self,
        # prompts: Optional[CustomPrompts] = None,
        llm: Optional[BaseLanguageModel] = None,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        vectorstore: Optional[VectorStore] = None,
        docs_chain_type: str = "stuff",
        docs_chain_kwargs: Optional[Dict] = None,
        configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            - prompts: dict of prompts to use for each chain type. If not given, default prompts will be used. Different sets of prompts are required for different chain types.
            For example, `stuff` chain_type requires `qa_prompt` and `condense_question_prompt` prompts, while `map_reduce` chain_type requires `condense_question_prompt`, `question_prompt` and `combine_prompt` prompts.
        """
        # prompts
        # if prompts is not None:
        #     _, self.docs_chain_kwargs = self._validate_docs_chain_and_prompts(
        #         prompts, docs_chain_type, docs_chain_kwargs
        #     )
        # else:
        #     self.condense_question_prompt = CONDENSE_QUESTION_TEMPLATE
        self.condense_question_prompt = (
            condense_question_prompt or CONDENSE_QUESTION_TEMPLATE
        )

        # llm for doc-chain
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
            output_key="answer",  # ☑️ required if return_source_documents=True
            return_messages=True,  # ☑️ required if return_source_documents=True
        )

        # build a chain with the given components
        self.chain = ConversationalRetrievalChain.from_llm(
            # https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/conversational_retrieval/base.py#L268
            # chain_type:
            # "stuff": default; to use all of the text from the documents in the prompt
            # "map_reduce": to batchify docs and feeds each batch with the question to LLM, and come up with the final answer based on the answers
            # "refine": to batchify docs and feeds the first batch to LLM, and then feeds the second batch with the answer from the first one, and so on
            # "map-rerank": to batchify docs and feeds each batch, return a score and come up with the final answer based on the scores
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            chain_type=docs_chain_type,
            condense_question_llm=self.condense_question_llm,
            condense_question_prompt=self.condense_question_prompt,
            combine_docs_chain_kwargs=docs_chain_kwargs,
            rephrase_question=True,  # default: True; Will pass the new generated question for retrieval
            return_source_documents=True,
            get_chat_history=None,  # default: None -> will use default;
            response_if_no_docs_found="잘 모르겠습니다.",
        )

    def __call__(self, question: str):
        return self.chain(question)

    # def _validate_docs_chain_and_prompts(
    #     self, prompts, docs_chain_type: str, docs_chain_kwargs: Dict
    # ):
    #     assert docs_chain_type in [
    #         "stuff",
    #         "map_reduce",
    #         "refine",
    #         "map-rerank",
    #     ], f"docs_chain_type must be one of ['stuff', 'map_reduce', 'refine', 'map-rerank'], but got {docs_chain_type}"

    #     if docs_chain_type == "stuff":
    #         assert (
    #             prompts.combine_prompt is None
    #             and prompts.collapse_prompt is None
    #             and prompts.refine_prompt is None
    #         )
    #         prompts["prompt"] = prompts.pop("qa_prompt")
    #     elif docs_chain_type == "map-rerank":
    #         assert (
    #             prompts.combine_prompt is None
    #             and prompts.collapse_prompt is None
    #             and prompts.refine_prompt is None
    #         )
    #         prompts["prompt"] = prompts.pop("qa_prompt")
    #     elif docs_chain_type == "refine":
    #         assert (
    #             prompts.refine_prompt
    #             and prompts.collapse_prompt is None
    #             and prompts.combine_prompt is None
    #         )
    #         prompts["question_prompt"] = prompts.pop("qa_prompt")
    #     else:
    #         assert (
    #             prompts.refine_prompt is None
    #             and prompts.collapse_prompt
    #             and prompts.combine_prompt
    #         )
    #         prompts["question_prompt"] = prompts.pop("qa_prompt")

    #     self.condense_question_prompt = prompts.pop("condense_question_prompt")
    #     docs_chain_kwargs.update(prompts)

    #     return prompts, docs_chain_kwargs

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
        llm: Optional[BaseLanguageModel] = None,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        # prompts: Optional[CustomPrompts] = None,
        docs_chain_type: str = "stuff",
        docs_chain_kwargs: Optional[Dict] = None,
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
            # prompts=prompts,
            llm=llm,
            vectorstore=vectorstore,
            condense_question_llm=condense_question_llm,
            condense_question_prompt=condense_question_prompt,
            docs_chain_type=docs_chain_type,
            docs_chain_kwargs=docs_chain_kwargs,
            configs=configs,
        )
