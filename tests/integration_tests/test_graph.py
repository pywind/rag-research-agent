import os
import uuid
from contextlib import contextmanager
from typing import Generator

import pytest
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain_redis import RedisConfig, RedisVectorStore
from langsmith import expect, unit

from src.index_graph import graph as index_graph
from src.retrieval_graph import graph
from src.shared.configuration import BaseConfiguration
from src.shared.retrieval import make_text_encoder

load_dotenv(override=True)


@contextmanager
def make_elastic_vectorstore(
    configuration: BaseConfiguration,
) -> Generator[VectorStore, None, None]:
    """Configure this agent to connect to a specific elastic index."""

    embedding_model = make_text_encoder(configuration.embedding_model)
    config = RedisConfig(
        index_name=os.environ["REDIS_INDEX_NAME"],
        redis_url=os.environ["REDIS_URL"],
    )
    vstore = RedisVectorStore(
        config=config,
        embeddings=embedding_model,
    )
    yield vstore


@pytest.mark.asyncio
@unit
async def test_retrieval_graph() -> None:
    simple_doc = 'In LangGraph, nodes are typically python functions (sync or async) where the first positional argument is the state, and (optionally), the second positional argument is a "config", containing optional configurable parameters (such as a thread_id).'
    config = RunnableConfig(
        configurable={
            "retriever_provider": "redis",
            "embedding_model": "openai/text-embedding-3-small",
            "model": "openai/gpt-4o-mini",
            "query_model": "openai/gpt-4o-mini",
        }
    )
    configuration = BaseConfiguration.from_runnable_config(config)

    doc_id = str(uuid.uuid4())
    result = await index_graph.ainvoke(
        {"docs": [{"page_content": simple_doc, "id": doc_id}]}, config
    )
    # print(result)
    expect(result["docs"]).against(lambda x: not x)  # we delete after the end
    # test general query
    res = await graph.ainvoke(
        {"messages": [("user", "Hi! How are you?")]},
        config,
    )
    expect(res["router"]["type"]).to_contain("general")
    # print(res)

    # test query that needs more info
    res = await graph.ainvoke(
        {"messages": [("user", "I am having issues with the tools")]},
        config,
    )
    # print(res)

    expect(res["router"]["type"]).to_contain("more-info")

    # test LangChain-related query
    res = await graph.ainvoke(
        {"messages": [("user", "What is a node in LangGraph?")]},
        config,
    )
    # print(res)

    expect(res["router"]["type"]).to_contain("langchain")
    response = str(res["messages"][-1].content)
    expect(response.lower()).to_contain("function")

    # clean up after test
    with make_elastic_vectorstore(configuration) as vstore:
        await vstore.adelete([doc_id])
