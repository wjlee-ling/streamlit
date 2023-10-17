from time import time
from typing import List, Union, Callable, Optional
from abc import abstractmethod
from langchain.schema import (
    BasePromptTemplate,
    Document,
    BaseDocumentTransformer,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BasePreprocessor:
    def __init__(
        self,
        prompt: BasePromptTemplate,
        llm: Union[BaseLanguageModel, BaseChatModel] = None,
    ):
        self.prompt = prompt
        self.llm = llm or ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_key="output",  # GPT 답변이 저장될 key; default='text'
        )

    @abstractmethod
    def preprocess(self, docs: List[Document], fn: Optional[Callable] = None):
        """Return a new list of Documents with preprocessed contents. Can apply `fn` to each 'page_content' before preprocessing.\
        refer: https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html"""
        return

    def _split(
        self, docs: List[Document], splitter: Optional[BaseDocumentTransformer] = None
    ):
        if splitter is None:
            splitter = RecursiveCharacterTextSplitter()
        return splitter.split_documents(docs)

    def preprocess_and_split(
        self,
        docs: List[Document],
        splitter: Optional[BaseDocumentTransformer] = None,
        fn: Optional[Callable] = None,
    ):
        start_preprocess = time()
        docs = self.preprocess(docs, fn)
        end_preprocess = time()
        print(
            f"☑️ Preprocessing took {(end_preprocess - start_preprocess):.3f} seconds for {len(docs)} document(s)."
        )
        start_split = time()
        docs = self._split(docs, splitter)
        end_split = time()
        print(
            f"☑️ Splitting took {(end_split - start_split):.3f} seconds with {len(docs)} newly split document(s)."
        )
        return docs


class PDFPreprocessor(BasePreprocessor):
    def preprocess(
        self, docs: List[Document], fn: Optional[Callable] = None
    ) -> List[Document]:
        """batch로 한번에 요청하여 빠르게 전처리. 다만 API 요청 제한이 있으므로 주의."""
        new_page_contents = self.chain.batch(
            list(
                map(
                    lambda doc: {
                        "doc": fn(doc.page_content) if fn else doc.page_content
                    },
                    docs,
                )
            )
        )  # returns: e.g. [{'doc': '안   녕 하세!요', 'output': '안녕하세요!'}, {'doc': '바보\n야 꺼~~~져!', 'output': '"바보야, 꺼져!"'}]
        # .apply 보다 2배 빠름

        for i, doc in enumerate(docs):
            doc.metadata["original_page_content"] = doc.page_content
            doc.page_content = new_page_contents[i]["output"]
        return docs
