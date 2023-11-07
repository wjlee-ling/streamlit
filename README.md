# ChatGPT-powered something w/ LangChain 

## How to Use

1. 데이터베이스를 새로 만들기

notion workspace에서 export 받아 unzip한 폴더를 load하여 데이터베이스 만들어야 함.

`python -m notion_bot --new --collection_name "{새 데이터베이스 이름}" --path "{unzip한 데이터 폴더 위치}" `

- 단축어 사용시

`python -m notion_bot -n -c "{새 데이터베이스 이름}" -p "{unzip한 데이터 폴더 위치}"`


2. (데이터베이스 존재 시) QA

`python -m notion_bot --collection_name "{기존 데이터베이스 이름}" --questions "{질문1} {질문2}"`

질문 갯수에는 제한이 없음.

- 단축어 사용시

`python -m notion_bot -c "{기존 데이터베이스 이름}" -q "{질문1} {질문2}"`

## Features
### 데이터
- 마크다운, csv 파일 읽어 데이터베이스화 (html 파일이나 이미지 처리 x)
- 워크스페이스 내 다른 문서를 언급(링크)한 문서를 검색(retrieval) 시 링크된 문서 내용을 포함(dereferencing)시켜 답변 생성

### W&B
- W&B 로 debugging 및 logging
- CLI로 아이디 입력하는 기능은 아직 도입 안 함
