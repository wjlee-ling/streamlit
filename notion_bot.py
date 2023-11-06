from modules.preprocessors import NotionPreprocessor, NotionDataLoader
from modules.notion import NotionBot
from modules.base import BaseBot
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


def run_notion_bot(args):
    # build a new collection for the local Chroma vectorstore and run the bot
    if args.new:
        loader = NotionDataLoader(path=args.path)
        preprocessor = NotionPreprocessor()
        bot = NotionBot.from_new_collection(
            loader=loader,
            preprocessor=preprocessor,
            collection_name="prompttown-1106",
        )
    else:
        # use an existing collection for the local Chroma vectorstore and run the bot
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0613",  # "gpt-4"
            temperature=0,
            verbose=True,
        )
        vectorstore = Chroma(
            persist_directory=f"db/chroma/{args.collection_name}",
            embedding_function=OpenAIEmbeddings(),
        )
        bot = NotionBot(
            llm=llm,
            vectorstore=vectorstore,
        )

    questions = ["담당자 이름은?", "프로젝트 기간은?", "고객사 분석한 거 요약해줘"]  #
    # 담당자의 이름은 김다혜입니다. 프로젝트 기간은 6월 첫째주부터 9월 첫째주까지입니다. 고객사 자료 분석은 6월 2일에 진행되었습니다. 담당자로는 남보름, 김성연, 김다혜가 참여하였습니다. 이 과정에서는 팀원 소개와 데이터바우처 사업에 대한 전반적인 이해를 진행하였으며, 고객사 미팅 전 질문리스트를 확정하였습니다. 현재는 요구사항 명세 작업을 진행 중이며, 프로젝트 자료 작성을 준비 중입니다.
    for question in questions:
        response = bot(question)
        print(response["answer"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--new",
        type=bool,
        default=False,
        help="Whether to create a new collection or use an existing one",
    )

    parser.add_argument(
        "-c",
        "--collection_name",
        type=str,
        default="prompttown-new",
        help="Name of the collection to use",
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="data/notion/DV-PromptTown",
        help="Path to the directory containing Notion files",
    )

    args = parser.parse_args()
    run_notion_bot(args)
