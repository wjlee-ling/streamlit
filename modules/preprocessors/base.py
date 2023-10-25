from time import time
from datetime import datetime
from pytz import timezone
from typing import List, Dict, Union, Callable, Optional
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
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)


class BasePreprocessor:
    def __init__(
        self,
        prompt: BasePromptTemplate,
        llm: Union[BaseLanguageModel, BaseChatModel] = None,
        splitter: Optional[BaseDocumentTransformer] = None,
    ):
        self.prompt = prompt
        self.llm = llm or ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            verbose=True,
        )
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_key="output",  # GPT 답변이 저장될 key; default='text'
        )
        self._splitter = splitter

    @abstractmethod
    def preprocess(
        self, docs: List[Document], fn: Optional[Callable] = None
    ) -> List[Document]:
        """Return a new list of Documents with preprocessed contents. Can apply `fn` to each 'page_content' before preprocessing.\
        refer: https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html"""
        pass

    @property
    @abstractmethod
    def splitter(self):
        pass

    def _split(self, docs: List[Document]):
        """
        MarkdownHeaderTextSplitter 등은 `split_documents` 메소드 대신 .split_text 호출해야함.
        """
        try:
            return self.splitter.split_documents(docs)
        except AttributeError:
            new_docs = []
            for doc in docs:
                new_docs.extend(self.splitter.split_text(doc))
            return new_docs

    def _get_current_time(self):
        seoul_tz = timezone("Asia/Seoul")
        curr_time = datetime.now(seoul_tz)
        return curr_time.strftime("%Y-%m-%d-%H-%M")

    def preprocess_and_split(
        self,
        docs: List[Document],
        fn: Optional[Callable] = None,
    ) -> List[Document]:
        curr_time = self._get_current_time()
        self.file = open(f"./splits_{curr_time}.txt", "w")

        start_preprocess = time()
        docs = self.preprocess(docs, fn)
        end_preprocess = time()
        print(
            f"☑️ Preprocessing took {(end_preprocess - start_preprocess):.3f} seconds for {len(docs)} document(s)."
        )

        start_split = time()
        docs = self._split(docs)
        end_split = time()
        print(
            f"☑️ Splitting into {len(docs)} newly split document(s) took {(end_split - start_split):.3f} seconds."
        )
        self.file.close()
        print(f"☑️ New splits saved to {self.file.name}.")
        return docs

    def save_output(self, output: Dict):
        from pprint import pprint

        pprint(output, stream=self.file)
