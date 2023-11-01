from modules.preprocessors import BasePreprocessor
from modules.templates_notion import TEMPLATE_NOTION_DEFAULT

import re
import glob
from typing import List, Tuple, Dict, Union, Callable, Optional
from pathlib import Path
from langchain.schema import (
    BasePromptTemplate,
    Document,
    BaseDocumentTransformer,
)
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import NotionDirectoryLoader, CSVLoader
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)


class NotionPreprocessor(BasePreprocessor):
    def __init__(
        self,
        splitter: Optional[BaseDocumentTransformer] = None,
        sub_splitter: Optional[BaseDocumentTransformer] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        super().__init__(splitter=splitter)
        self._splitter = splitter
        self.chunk_size = chunk_size
        self.sub_splitter = sub_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def splitter(self):
        if self._splitter is None:
            return MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "#")])
        else:
            return self._splitter

    def _get_file_path(self, filename: str, directory: str = None):
        directory = "data/notion/DV-PromptTown"  # 💥 TODO: guess from the full path
        files = glob.glob(f"{directory}/**/*{filename}", recursive=True)
        return files

    def _extract_doc_id(self, doc: Document):
        """Notion이 자동으로 붙인 파일/폴더 id (소문자 + 숫자 + (.확장자)) 추출."""
        return doc.metadata["source"].split(" ")[-1]

    def _handle_links(
        self,
        doc: Document,
        file_formats: Tuple[str] = ("md", "csv"),
    ) -> Document:
        """마크다운 문서에서 포함된 하이퍼링크 스트링이 임베딩 되지 않게 "(link@{num})"으로 변환하고 메타데이터 리스트(인덱스{num})에 저장 (key는 'links')"""
        page_content = doc.page_content
        page_content_to_process = doc.page_content
        doc.metadata["links"] = []
        while match := re.search(
            r"(?<=\])\(%[A-Za-z0-9\/\(\)%\.~]+",
            page_content_to_process,
        ):
            (match_start_idx, non_match_start_idx) = match.span()
            page_content_to_process = page_content[non_match_start_idx:]
            if match.group().strip(")]}").endswith(file_formats):
                # 링크 스트링 메타 데이터에 추가
                doc.metadata["links"].append(match.group().strip("()"))

                # 링크 스트링 삭제
                page_content = (
                    page_content[:match_start_idx]
                    + f"(link@{len(doc.metadata['links'])-1})"
                    + page_content[non_match_start_idx:]
                )

            else:
                ## .png 등은 그냥 링크 삭제
                page_content = (
                    page_content[:match_start_idx] + page_content[non_match_start_idx:]
                )
        doc.page_content = page_content

        return doc

    def _split_by_len(self, chunk: str) -> List[str]:
        """self.chunk_size 보다 길면 sub_splitter로 split"""
        if len(chunk) > self.chunk_size:
            return self.sub_splitter.split_text(chunk)
        else:
            return [chunk]

    def _split(
        self,
        doc: Document,
    ) -> List[Document]:
        """
        1. `MarkdownHeaderTextSplitter`는 `str`의 doc 하나만 처리 가능하므로 처리 후 기존 메타데이터 추가
        2. 메타데이터로 들어간 헤더 정보 추가
        3. `MarkdownHeaderTextSplitter`로 자른 결과가 너무 길 때 (원 문서에 헤더가 없어서) self.chunk_size 만큼 추가로 자름
        """
        original_metadata = doc.metadata
        chunks = self.splitter.split_text(doc.page_content)  # split by headers
        new_chunks = []
        for chunk in chunks:
            for header_level, header_content in chunk.metadata.items():
                chunk.page_content = (
                    f"{header_level} {header_content}\n{chunk.page_content}"
                )
            splits_within_max_len = self._split_by_len(chunk.page_content)
            new_chunks.extend(
                [
                    Document(
                        page_content=split, metadata={**original_metadata}
                    )  ## "source" 와 "links" 유지
                    for split in splits_within_max_len
                ]
            )
        return new_chunks

    def preprocess_and_split(
        self,
        docs: List[Document],
        fn: Optional[Callable] = None,
    ) -> List[Document]:
        new_chunks = []
        for doc in docs:
            # 본문에 포함된 링크를 placeholder와 바꾸고, 메타데이터로 옮김
            doc = self._handle_links(doc)
            chunks = self._split(doc)
            new_chunks.extend(chunks)
        new_chunks = self._aggregate_chunks(new_chunks)
        self.save_output(new_chunks)

        return new_chunks

    def _aggregate_chunks(
        self,
        chunks: List[Document],
    ) -> List[Document]:
        """
        `TextSplitter`가 자른 문서가 1. 길이가 너무 짧고 2. 같은 부모 디렉토리를 갖을 때 합치기 (+ 메타데이터 소스 수정)
        """
        if len(chunks) == 1:
            return chunks

        prev_chunk = None
        new_chunks = []
        for chunk in chunks:
            if prev_chunk is None:
                # 맨 처음 chunk는 바로 prev_chunk
                prev_chunk = chunk
                continue
            if (
                len(prev_chunk.page_content) + len(chunk.page_content) < 500
                and prev_chunk.metadata["source"].split("/")[:-1]
                == chunk.metadata["source"].split("/")[:-1]
            ):
                chunk.page_content = prev_chunk.page_content + "\n" + chunk.page_content
                ## 동일 부모 폴더의 파일들을 합치는 경우 "부모폴더/파일1&&파일2" 로 메타 데이터 저장
                chunk.metadata["source"] = (
                    prev_chunk.metadata["source"]
                    + "&&"
                    + chunk.metadata["source"].split("/")[-1]
                )
            else:
                new_chunks.append(prev_chunk)

            prev_chunk = chunk

        if prev_chunk != new_chunks[-1]:
            new_chunks.append(prev_chunk)

        return new_chunks


class NotionDataLoader(BaseLoader):

    """
    Notion 데이터 폴더의 markdown 파일과 csv 파일들을 로드. 현재 `NotionDirectoryLoader`는 디렉토리의 `md` 파일만
    `CSVLoader`는 기타 csv 파일을 읽음
    일단 현재(11/1)는 '_all'의 접미사가 붙지 않은 하위 페이지 포함하지 않은 데이터베이스만 읽음
    """

    def __init__(self, path: str, *, encoding: str = "utf-8-sig") -> None:
        self.encoding = encoding
        self.path = path
        self.MD_Loader = None
        self.CSV_Loader = None

    def _load_markdown(self) -> List[Document]:
        self.MD_Loader = NotionDirectoryLoader(path=self.path, encoding=self.encoding)
        return self.MD_Loader.load()

    def _load_csv(self) -> List[Document]:
        """
        Load csv files that do not have a corresponding `_all.csv` file, meaning they don't contain embedded sub-pages
        """
        csv_files = list(Path(self.path).rglob("*.csv"))
        csv_files = [
            file
            for file in csv_files
            if not file.with_stem(f"{file.stem}_all").exists()
        ]
        docs = []
        for csv_file in csv_files:
            self.CSV_Loader = CSVLoader(file_path=csv_file, encoding=self.encoding)
            docs.extend(self.CSV_Loader.load())
        return docs

    def load(self):
        markdown_docs = self._load_markdown()
        csv_docs = self._load_csv()
        return markdown_docs + csv_docs
