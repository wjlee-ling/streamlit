from modules.preprocessors.base import BasePreprocessor

from typing import List, Dict, Callable, Optional
from langchain.schema import Document


class PDFPreprocessor(BasePreprocessor):
    def get_default_fn(
        self,
        doc: Document,
        previous_doc_content: List = [None],
    ) -> Dict:
        page_content = doc.page_content

        if previous_doc_content[0] is None:
            context = ""
        else:
            context = "\n".join(previous_doc_content[0].strip().split("\n")[-2:])

        previous_doc_content[0] = page_content

        return {
            "context": context,
            "doc": page_content,
        }

    def preprocess(
        self,
        docs: List[Document],
        fn: Optional[Callable[List[Document], Dict]] = None,
    ) -> List[Document]:
        """batch로 한번에 요청하여 빠르게 전처리. 다만 API 요청 제한이 있으므로 주의."""

        fn = fn or self.get_default_fn
        new_page_contents = self.chain.batch(
            list(
                map(
                    lambda doc: fn(doc),
                    docs,
                )
            )
        )  # returns: e.g. [{'doc': '안   녕 하세!요', 'output': '안녕하세요!'}, {'doc': '바보\n야 꺼~~~져!', 'output': '"바보야, 꺼져!"'}]
        # .apply 보다 2배 빠름

        # new_docs = []
        # with open("output_1018_with_prev.txt", "w") as f:
        for i, doc in enumerate(docs):
            # f.write(f"\n=============== num: {str(i)} ===================\n")
            # f.write(doc.page_content)
            # f.write("-------------- revision ----------------\n")
            # f.write(new_page_contents[i]["output"])
            doc.metadata["original_page_content"] = doc.page_content
            doc.page_content = new_page_contents[i]["output"]

        return docs
