from modules.base import BaseBot
from modules.templates import (
    PDF_PREPROCESS_TEMPLATE,
    PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
)
from modules.preprocessors import PDFPreprocessor

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import MarkdownTextSplitter


PATH = "data/test/[제안요청서]AIBanker서비스구축을위한데이터셋구축(우리은행)_F.pdf"
pdf_loader = PDFPlumberLoader(
    PATH,
)
preprocessor = PDFPreprocessor(
    prompt=PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
)

bot = BaseBot.from_new_collection(
    loader=pdf_loader,
    collection_name="woori_pdf_prev_md",
    preprocessor=preprocessor,
)
