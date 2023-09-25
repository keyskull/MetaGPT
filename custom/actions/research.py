#!/usr/bin/env python

from __future__ import annotations

import asyncio
import json
from typing import Callable

from pydantic import parse_obj_as
import sys

from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.tools.search_engine import SearchEngine
from metagpt.tools.web_browser_engine import WebBrowserEngine, WebBrowserEngineType
from metagpt.utils.text import generate_prompt_chunk, reduce_message_length
from langchain.vectorstores.base import VectorStore
import dirtyjson
LANG_PROMPT = "Please respond in {language}."

RESEARCH_BASE_SYSTEM = """You are an AI critical thinker research assistant. Your sole purpose is to write well \
written, critically acclaimed, objective and structured reports on the given text."""

RESEARCH_TOPIC_SYSTEM = "You are an AI researcher assistant, and your research topic is: {topic}"

SEARCH_TOPIC_PROMPT = """Please provide only 10 necessary keywords related to your research topic for Google search. \
Your response must be in JSON format, for example: {"keyword1": "value1", "keyword2" : "value2"}."""

SUMMARIZE_SEARCH_PROMPT = """
### Requirements
- The keywords related to your research topic and the search results are shown in the "Search Result Information" section.
- Generate questions which are related to your research topic base on the search results.

### Search Result Information
{search_results}


Your response must be in JSON Format with only {decomposition_nums} Google search questions, for example if you have two questions:
{{ 
    "question1": "",
    "question2": ""
}}

"""

COLLECT_AND_RANKURLS_PROMPT = """
### Requirements
- Sort the remaining search result array based on the link credibility. If two results have equal credibility, prioritize them based on the relevance. 



### Topic
{topic}
### Query
{query}
### The search results
{results}


Your reponse must be in JSON Format with only ranked search results, for example: 
{{
    "rank1": "index1",
    "rank1": "index2"
}}

"""

WEB_BROWSE_AND_SUMMARIZE_PROMPT = '''

### Reference Information
{content}

### Requirements
- Utilize the text in the "Reference Information" section to respond to the question "{query}".
- If the question cannot be directly answered using the text, but the text is related to the research topic, please provide \
a comprehensive summary of the text.
- If the text is entirely unrelated to the research topic, please reply with a simple text "Not relevant." only, otherwise plsease exclude it.
- Include all relevant factual information, numbers, statistics, etc., if available.
'''


CONDUCT_RESEARCH_PROMPT = '''### Reference Information
{content}

### Requirements
Please provide a detailed research report in response to the following topic: "{topic}", using the information provided \
above. The report must meet the following requirements:

- Focus on directly addressing the chosen topic.
- Ensure a well-structured and in-depth presentation, incorporating relevant facts and figures where available.
- Present data and findings in an intuitive manner, utilizing feature comparative tables, if applicable.
- The report should have a minimum word count of 2,000 and be formatted with Markdown syntax following APA style guidelines.
- Include all source URLs in APA format at the end of the report.
'''


class CollectLinks(Action):
    """Action class to collect links from a search engine."""
    def __init__(
        self,
        name: str = "",
        *args,
        rank_func: Callable[[list[str]], None] | None = None,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.desc = "Collect links from a search engine."
        self.search_engine = SearchEngine()
        self.rank_func = rank_func

    async def run(
        self,
        topic: str,
        decomposition_nums: int = 4,
        url_per_query: int = 4,
        system_text: str | None = None,
    ) -> dict[str, list[str]]:
        """Run the action to collect links.

        Args:
            topic: The research topic.
            decomposition_nums: The number of search questions to generate.
            url_per_query: The number of URLs to collect per search question.
            system_text: The system text.

        Returns:
            A dictionary containing the search questions as keys and the collected URLs as values.
        """
        system_text = system_text if system_text else RESEARCH_TOPIC_SYSTEM.format(topic=topic)
        # keywords = await self._aask(SEARCH_TOPIC_PROMPT, [system_text])
        # keywords = await self._aask(SEARCH_TOPIC_PROMPT + "\n" + system_text)
        # try:
        #     keywords = dirtyjson.loads(keywords)
        #     keywords = [ i for i in keywords.values()]

        #     # keywords = parse_obj_as(list[str], keywords)
        # except Exception as e:
        #     logger.exception(f"fail to get keywords related to the research topic \"{topic}\" for {e}")
        keywords = [topic]
        results = await asyncio.gather(*(self.search_engine.run(i, as_string=False) for i in keywords))

        def gen_msg():
            while True:
                search_results = "\n".join(f"#### Keyword: {i}\n Search Result: {j}\n" for (i, j) in zip(keywords, results))
                prompt = SUMMARIZE_SEARCH_PROMPT.format(decomposition_nums=decomposition_nums, search_results=search_results)
                yield prompt
                remove = max(results, key=len)
                remove.pop()
                if len(remove) == 0:
                    break
        prompt = reduce_message_length(gen_msg(), self.llm.model, system_text, CONFIG.max_tokens_rsp)
        logger.info(prompt)
        queries = await self._aask(prompt, [topic])
        logger.info(queries)
        try:
            queries = queries.replace("```","")
            queries = queries.replace("```json","")
            queries = dirtyjson.loads(queries)
            queries = [ i for i in queries.values()]

            # queries = json.loads("{" + queries + "}")
            # queries = parse_obj_as(list[str], queries)
        except Exception as e:
            logger.exception(f"fail to break down the research question due to {e}")
            sys.exit(0) 
            queries = keywords
        ret = {}
        from langchain.vectorstores import FAISS
        from langchain.vectorstores.faiss import dependable_faiss_import
        from langchain.docstore import InMemoryDocstore 
        from langchain.embeddings import OpenAIEmbeddings

        
        embedding = OpenAIEmbeddings(
            # deployment=CONFIG.openai_api_model,
            # tiktoken_model_name="gpt-4",
            openai_api_version="2020-11-07",
            show_progress_bar=True)
        store = FAISS.from_texts(texts="initial text", embedding = embedding)

        
        await self._search_urls(queries, store, num_results = url_per_query)

        links = []
        for query in queries:
            (result, lnks) = await self._rank_urls(topic, query, store, links)
            ret[query] = result
            links.append(lnks)
        logger.info(ret)
        return ret

    async def _search_urls(self, queries: list[str], store: VectorStore, num_results: int = 4):
        max_results = max(num_results * 2, 6)
        for query in queries:
            results = await self.search_engine.run(query, max_results=max_results, as_string=False)
            _results =[r["title"] + "\n" + r["snippet"] for r in results]
            store.add_texts(_results, metadatas=results, show_progress_bar=True)


    async def _rank_urls(self, topic: str, query: str, store: VectorStore, links: list[str],  num_results: int = 4) -> tuple[list[str],str]:
        """Search and rank URLs based on a query.

        Args:
            topic: The research topic.
            query: The search query.
            num_results: The number of URLs to collect.

        Returns:
            A list of ranked URLs.
        """

        prompt = topic + "\n" + query
        docs = store.similarity_search_with_score(prompt, score_threshold=1)

        docs = [(d, s) for (i, (d,s)) in enumerate(docs) ]
        docs.sort(key=lambda x: x[1], reverse=True)
        ndocs = []
        for i,s in docs:
            if i.metadata["link"] not in links:
                links.append(i.metadata["link"])
                ndocs.append(i)
        docs = ndocs
        results = [i.metadata for i in docs]
        if self.rank_func:
            results = self.rank_func(results)

        return ([i["link"] for i in results[:num_results]], links)


class WebBrowseAndSummarize(Action):
    """Action class to explore the web and provide summaries of articles and webpages."""
    def __init__(
        self,
        *args,
        browse_func: Callable[[list[str]], None] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if CONFIG.model_for_researcher_summary:
            self.llm.model = CONFIG.model_for_researcher_summary
        self.web_browser_engine = WebBrowserEngine(
            engine=WebBrowserEngineType.CUSTOM if browse_func else None,
            run_func=browse_func,
        )
        self.desc = "Explore the web and provide summaries of articles and webpages."

    async def run(
        self,
        url: str,
        *urls: str,
        query: str,
        system_text: str = RESEARCH_BASE_SYSTEM,
    ) -> dict[str, str]:
        """Run the action to browse the web and provide summaries.

        Args:
            url: The main URL to browse.
            urls: Additional URLs to browse.
            query: The research question.
            system_text: The system text.

        Returns:
            A dictionary containing the URLs as keys and their summaries as values.
        """
        logger.info(url)
        logger.info(urls)
        contents = await self.web_browser_engine.run(url, *urls)
        if not urls:
            contents = [contents]

        summaries = {}
        prompt_template = WEB_BROWSE_AND_SUMMARIZE_PROMPT.format(query=query, content="{}")
        for u, content in zip([url, *urls], contents):
            content = content.inner_text
            chunk_summaries = []
            for prompt in generate_prompt_chunk(content, prompt_template, self.llm.model, system_text, CONFIG.max_tokens_rsp):
                logger.info(prompt)
                summary = await self._aask(prompt, [system_text])
                if "not relevant" in summary.lower():
                    continue
                chunk_summaries.append(summary)

            if not chunk_summaries:
                summaries[u] = None
                continue

            if len(chunk_summaries) == 1:
                summaries[u] = chunk_summaries[0]
                continue

            content = "\n".join(chunk_summaries)
            prompt = WEB_BROWSE_AND_SUMMARIZE_PROMPT.format(query=query, content=content)
            logger.info(prompt)
            summary = await self._aask(prompt, [system_text])
            summaries[u] = summary
        return summaries


class ConductResearch(Action):
    """Action class to conduct research and generate a research report."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if CONFIG.model_for_researcher_report:
            self.llm.model = CONFIG.model_for_researcher_report

    async def run(
        self,
        topic: str,
        content: str,
        system_text: str = RESEARCH_BASE_SYSTEM,
    ) -> str:
        """Run the action to conduct research and generate a research report.

        Args:
            topic: The research topic.
            content: The content for research.
            system_text: The system text.

        Returns:
            The generated research report.
        """
        prompt = CONDUCT_RESEARCH_PROMPT.format(topic=topic, content=content)
        logger.debug(prompt)
        self.llm.auto_max_tokens = True
        return await self._aask(prompt, [system_text])


def get_research_system_text(topic: str, language: str):
    """Get the system text for conducting research.

    Args:
        topic: The research topic.
        language: The language for the system text.

    Returns:
        The system text for conducting research.
    """
    return " ".join((RESEARCH_TOPIC_SYSTEM.format(topic=topic), LANG_PROMPT.format(language=language)))
