from modules.base import BaseBot
from modules.templates import (
    PDF_PREPROCESS_TEMPLATE,
    PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
)
from modules.preprocessors import PDFPreprocessor

from langchain.document_loaders import PyPDFDirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import MarkdownTextSplitter


PATH = "data/test/[제안요청서]AIBanker서비스구축을위한데이터셋구축(우리은행)_F.pdf"
pdf_loader = PDFPlumberLoader(
    PATH,
)
preprocessor = PDFPreprocessor(
    prompt=PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
    splitter=MarkdownTextSplitter(),
)

bot = BaseBot.from_new_collection(
    loader=pdf_loader,
    collection_name="woori_pdf_prev_md",
    preprocessor=preprocessor,
)
print(bot("우리은행이 진행하는 사업명은?"))
print(bot("AI Banker 학습 데이터를 구축하려고 하는 회사는 어디인가?"))
print(bot("사업 제안서는 어디다 제출하면 되나요?"))
print(bot("제안 설명회는 언제 진행되나요?"))
print(bot("제안서 관련 문의처는?"))
