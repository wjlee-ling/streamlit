"""
chain 타입에 따라 template의 포맷과 variable이 달라짐.
`ConversationalRetrievalChain`의 경우, quenstion-condensing을 위한 `CONDENSE_QUESTION_PROMPT` template과 answer-generation을 위한 `QA_PROMPT` template이 필요함. (https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/conversational_retrieval/prompts.py#L10)
`CONDENSE_QUESTION_PROMPT` template의 경우, `chat_history`와 `question`을 variable로, `QA_PROMPT`의 경우 `question`과 `context`를 variable로 사용함.

"""

rules = """[규칙] \
1. [답변]은 온전한 문장으로 작성하라. \
2. [답변]은 한국어로 작성하라. \
3. [답변]은 반말은 사용하지 말고 존댓말로 격식있게 하라. \
4. [답변]을 모르겠다면 "잘 모르겠습니다."라고 작성하라. \
5. [답변]에 욕, 비속어, 인종차별, 성차별 기타 소수자 혐오 발언은 하지 말라.\
"""

template_en = """Observe the following rules to answer the question at the end.\
    1. Answer the question in a complete sentence.\
    2. Answer in Korean.\
    3. Answer in a polite manner with honorifics. \
    4. If you don't know the answer, just type "잘 모르겠습니다".\
    5. DO NOT swear or use offensive language.\
    Given the rules, the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    chat history: {chat_history}\
    question: {question}\
    answer:"""

template_kor = """{rules} \
[대화내역] {chat_history} \
[질문] {question} \
위 [규칙]과 [대화내역]을 참고하여 [질문]에 대한 [답변]을 작성하라. \
[답변] """
