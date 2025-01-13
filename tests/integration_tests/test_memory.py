import uuid

import langsmith as ls
import pytest
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
from langgraph_sdk import get_client

from src.config.model import User
from src.memory.graph import builder
from src.memory.utils import create_memory_function
from src.retrieval_graph.configuration import AgentConfiguration

load_dotenv(override=True)


@pytest.mark.asyncio
@ls.unit
async def test_patch_memory_stored():
    mem_store = InMemoryStore()
    graph = builder.compile(store=mem_store)
    thread_id = str(uuid.uuid4())
    user_id = "henry-vu"
    config = {
        "configurable": {"memory_types": [create_memory_function(User)]},
        "thread_id": thread_id,
        "user_id": user_id,
    }
    await graph.ainvoke(
        {"messages": [("user", "My name is Henry. I like fun things")]}, config
    )
    namespace = (user_id, "user_states")
    memories = mem_store.search(namespace)
    # print(f"memories: {memories}")

    ls.expect(len(memories)).to_be_greater_than(0)
    mem = memories[0]
    ls.expect(mem.value.get("preferred_name")).to_contain("Henry")

    await graph.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "Even though your name is Mitty, I prefer to call you with that name.",
                )
            ]
        },
        config,
    )
    memories = mem_store.search(namespace)
    # print(f"memories: {memories}")

    ls.expect(len(memories)).to_be_greater_than(0)
    mem = memories[0]

    # Check that searching by a different namespace returns no memories
    bad_namespace = ("user_states", "my-bad-test-user", "User")
    memories = mem_store.search(bad_namespace)
    ls.expect(memories).against(lambda x: not x)


@pytest.mark.asyncio
@ls.unit
async def _test_schedule_memories() -> None:
    """Prompt the bot to respond to the user, incorporating memories (if provided)."""
    thread_id = "b64f787d-f1a9-4daf-9316-d09ba1e04bbc"

    config = {
        "user_id": "henry-vu",
        "delay_seconds": 10,
        "mem_assistant_id": "memory",
    }
    configurable = AgentConfiguration(**config)
    memory_client = get_client(url="http://localhost:2024")

    # Add validation for mem_assistant_id
    if not configurable.mem_assistant_id:
        raise ValueError("Memory assistant ID is not configured")

    await memory_client.runs.create(
        thread_id=thread_id,
        multitask_strategy="enqueue",
        after_seconds=config["delay_seconds"],
        assistant_id=config["mem_assistant_id"],  # Make sure this ID exists and is valid
        input={"messages": []},
        config={
            "configurable": {
                "user_id": config["user_id"],
                "memory_types": configurable.memory_types,
            },
        },
    )

    # print(f"output: {output}")
