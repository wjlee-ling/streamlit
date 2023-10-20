from modules.templates import QUESTION_ANSWER_AUGMENTATION_TEMPLATE
from langchain.chat_models import ChatOpenAI


class Augmentator:
    def __init__(self, llm=None, prompt=None):
        self.llm = llm or ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
        )
        self.prompt = prompt or QUESTION_ANSWER_AUGMENTATION_TEMPLATE

    def augment(self, question, answer):
        return self.llm(self.prompt.format_messages(question=question, answer=answer))


augmentator = Augmentator()
results = augmentator.augment(
    "회사에서 지인 추천 제도를 이용하려면?", "지인 추천 제도를 이용하려면 TEXTNET 홈페이지의 채용 페이지에서 진행 상황을 확인하세요."
)
print(results)
