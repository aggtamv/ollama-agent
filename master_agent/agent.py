import os
from dotenv import load_dotenv
from .prompts import SYSTEM_TEMPLATE_PROMPT
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
import yaml
from .tools import (
    read_csv,
    write_output,
    create_classifier,
    execute_python
)

load_dotenv()


def get_available_agent(user_input):
    tools = [
        read_csv, 
        write_output,
        create_classifier,
        execute_python
    ]

    llm = ChatOllama(
            model='mistral',
            temperature=0,
        )

    llm.bind_tools(tools=tools)

    agent_memory = MemorySaver()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=SYSTEM_TEMPLATE_PROMPT),
        checkpointer=agent_memory
    )