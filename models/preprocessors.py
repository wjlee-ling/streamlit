from typing import List, Union, Callable, Optional
from abc import abstractmethod
from langchain.schema import (
    BasePromptTemplate,
    Document,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


class BasePreprocessor:
    def __init__(
        self,
        prompt: BasePromptTemplate,
        llm: Union[BaseLanguageModel, BaseChatModel] = None,
    ):
        self.prompt = prompt
        self.llm = llm or ChatOpenAI(model_name="gpt-4")
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    @abstractmethod
    def preprocess(self, docs: List[Document], fn: Optional[Callable] = None):
        """Return a new list of Documents with preprocessed contents. Can apply `fn` to each 'page_content' before preprocessing."""
        return


class PDFPreprocessor(BasePreprocessor):
    def preprocess(self, docs: List[Document], fn: Optional[Callable] = None) -> str:
        preprocessed_docs = []
        for doc in docs:
            content = fn(doc.page_content) if fn else doc.page_content
            content = self.chain.run(content)
            doc.metadata["original_page_content"] = doc.page_content
            doc.page_content = content
            preprocessed_docs.append(doc)
            from pprint import pprint

            pprint(doc)

        return preprocessed_docs
