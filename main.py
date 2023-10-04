import streamlit as st
from qa.retriever import QARetriever 
from streamlit import session_state as sst

bot = QARetriever(url="https://textnet.kr/about")

if "messages" not in sst:
    sst.messages = []

st.title("Prompt Engineering Bot")

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
    answer = response["result"]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})