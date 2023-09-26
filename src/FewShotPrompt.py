"""
UX라이팅 원칙이 있는 CSV파일을 읽어 few-shot learning을 수행
"""
import pandas as pd
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

template = PromptTemplate.from_template("""UX 라이팅(UX writing) 가이드라인을 작성해주세요. \
가이드라인은 최소 1개 이상의 UX 라이팅 원칙에 기반해야 하며, 그 원칙은 가독성, 가시성, 명확성, 유저 사용성, 일관성, 유연성 등입니다. 아래 예시들을 참고하여 새 UX라이팅 원칙을 가능한 많이 작성해주세요.\n""")
example_template = PromptTemplate(
    template="[가이드라인]: {guideline} [가이드라인 설명]: {description} [원칙]:{core} [예시]:[don't]: {dont} [do]: {do}",
    input_variables=['guideline', 'core', 'description', 'dont', 'do'])

df = pd.read_csv('data/UXWriting_guidelines_0925.csv', encoding='utf-8')
examples = df.to_dict('records')

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    suffix="{input}",
    input_variables=['input'],
)

output = prompt.format(input="UX 라이팅 가이드라인과 해당되는 원칙을 최대한 많이 json 형식으로  작성해주세요.")
print(output)