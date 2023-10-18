__import__("pysqlite3")

import sys, io, os

if "pysqlite3" in sys.modules:
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from modules.base import BaseBot
from modules.compare import CompareBot
from modules.templates import (
    PDF_PREPROCESS_TEMPLATE,
    PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
)
from modules.preprocessors import PDFPreprocessor

import chromadb
from pprint import pprint
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
from streamlit import session_state as sst

PATH = "./user_data.pdf"
COLLECTION_NAME = "woori_pdf_prev_md"
COLLECTION_NAME_COMPARE = "compare"


@st.cache_resource
def init_embeddings(name="openai"):
    embedidngs = OpenAIEmbeddings()
    return embedidngs


@st.cache_resource
def init_bot(collection_name, _embeddings):
    client = chromadb.PersistentClient("db/chroma/woori_pdf_prev_md")
    # collection = client.get_collection(name=collection_name)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=_embeddings,
    )
    bot = BaseBot(
        vectorstore=vectorstore,
    )
    # if not os.path.exists(path):
    #     bot = BaseBot.from_new_collection(
    #         loader=PDFPlumberLoader(path),
    #         preprocessor=PDFPreprocessor(
    #             prompt=PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
    #         ),
    #         collection_name=collection_name,
    #     )
    # else:
    #     DB_DIR = "db/chroma/"
    #     client_settings = chromadb.config.Settings(
    #         chroma_db_impl="duckdb+parquet",
    #         persist_directory=DB_DIR,
    #         anonymized_telemetry=False,
    #     )

    #     vectorstore = Chroma(
    #         collection_name=collection_name,
    #         embedding_function=embeddings,
    #         client_settings=client_settings,
    #         persist_directory=DB_DIR,
    #     )
    #     bot = BaseBot(
    #         vectorstore=vectorstore,
    #         collection_name=collection_name,
    #     )
    print(bot)
    return bot


@st.cache_resource
def init_compare_bot(collection_name, _embeddings):
    client = chromadb.PersistentClient("db/chroma/compare")
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=_embeddings,
    )
    bot = CompareBot(
        vectorstore=vectorstore,
    )
    return bot


@st.cache_data
def get_info():
    return """
    문서 내용, 포맷별 특성에 맞는 세팅과 전처리 모듈, 프롬프트를 사용합니다. 현재 데모에서 읽은 PDF 문서는 우리은행 사업 관련 제안요청서입니다.
    """


st.title("GPT-Powered Chat Bot")
info = get_info()
embeddings = init_embeddings()

if "bot" not in sst:
    sst.bot = init_bot(COLLECTION_NAME, embeddings)
if "compare_bot" not in sst:
    sst.comp_bot = init_compare_bot(COLLECTION_NAME_COMPARE, embeddings)

if "messages" not in sst:
    sst.messages = []
    sst.comp_messages = []


with st.expander("Info"):
    st.info(info)
    st.write("**비교 예시**")
    st.image("public/예시1.png")
#     col1, col2 = st.columns(2)
#     bad_ex = "`- 16 -붙임 제안참가 서약서2【 】 \n제안참가 서약서\n \n제안 건 명 우리은행 언어모델 : ( ) AI Banker ■ 의 학습 데이터셋 구축\n \n우리은행 AI Banker언어모델의 학습 데이터셋 구축 사업 제안에 참여할 기회를 주신 \n것을 감사드리며 우리은행과 제안사 상호간에 준수할 기본 사항이 다음과 같음을 확 , \n인하고 이를 이행하겠습니다.`"
#     good_ex = """## [붙임 2] 제안참가 서약서
# ### 제안참가 서약서
# #### 제안 건 명: (우리은행) AI Banker 언어모델의 학습 데이터셋 구축
# 우리은행 AI Banker 언어모델의 학습 데이터셋 구축 사업 제안에 참여할 기회를 주신 것을 감사드리며, 우리은행과 제안사 상호간에 준수할 기본 사항이 다음과 같음을 확인하고 이를 이행하겠습니다.
#     """
#     with col1:
#         st.write("**디폴트 세팅값으로 생성한 DB**")
#         st.write(bad_ex)
#     with col2:
#         st.write("**조정한 세팅값과 전처리 모듈을 추가해 생성한 DB**")
#         st.write(good_ex)

# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
uploaded_file = 1
if uploaded_file is not None:
    # with open(PATH, "wb") as f:
    #     f.write(uploaded_file.read())

    for message in sst.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # for message in sst.comp_messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["compare"])

    if prompt := st.chat_input("무엇이든 물어보세요"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        sst.messages.append({"role": "user", "content": prompt})
        sst.comp_messages.append({"role": "user", "content": prompt})

        # Get assistant response
        response = sst.bot(
            prompt
        )  # keys: [question, chat_history, answer, source_documents]
        print(response)
        answer = response["answer"]
        source = response["source_documents"][0]
        source_content = source.page_content  # .replace("\n", " ")
        source_src = source.metadata["source"]
        source_page = source.metadata["page"]

        # compare versions
        comp_response = sst.comp_bot(
            prompt
        )  # keys: [question, chat_history, answer, source_documents]
        print(response)
        comp_answer = comp_response["answer"]
        # comp_source = comp_response["source_documents"][0]
        # comp_source_content = comp_source.page_content  # .replace("\n", " ")
        # comp_source_src = comp_source.metadata["source"]
        # comp_source_page = comp_source.metadata["page"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
            if "죄송" not in answer or "정보가 제공되지" not in answer or "찾을 수 없" in answer:
                # output = io.StringIO()
                # pprint(source_content, stream=output)
                # output_string = output.getvalue()
                with st.expander("참고한 문서 예시"):
                    st.info(f"주요 출처 페이지: {source_page}")
                    st.markdown(source_content)

                with st.expander("cf. 기본 세팅값 + 전처리 없는 모델의 답변"):
                    st.markdown(comp_answer)

                # st.info(f"출처 문서: {output_string}\n\n출처 링크: {source_src}")
            # Add assistant response to chat history
            sst.messages.append({"role": "assistant", "content": answer})
