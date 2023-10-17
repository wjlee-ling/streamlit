from modules.preprocessors.base import BasePreprocessor

from typing import List, Callable, Optional
from langchain.schema import Document


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
