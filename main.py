__import__('pysqlite3')

import streamlit as st
from qa.retriever import QARetriever 
from streamlit import session_state as sst

import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

@st.cache_resource
def get_bot(url="https://textnet.kr/about"):
    return QARetriever(url=url)

@st.cache_data
def get_info():
    return """
    이 챗봇은 'https://textnet.kr/about'페이지 기반으로 답변합니다.

    **[사용 예시]**\n
    Q: 회사 대표 이름은\n
    A: 회사의 대표 이름은 고경민입니다.\n
    Q: 회사 주소는\n
    A: 회사의 주소는 서울시 용산구 한강대로 366 트윈시티 남산 오피스동 패스트파이브 서울역점 807호, 812호입니다.\n

    **[TO-DO]**
    - [ ] 더 많은 문서를 이용하여 임베딩 & DB에 저장
    - [ ] question-condensing w/ `ConversationalRetrievalChain`
    """

bot = get_bot()
info = get_info()

st.title("Prompt Engineering Bot")
with st.expander("Info"):    
    st.write(info)

if "messages" not in sst:
    sst.messages = []


for message in sst.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("무엇이든 물어보세요"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    response = bot(prompt)
    answer = response["answer"]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
