import logging
from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet-latest"
prompt_base = """You are a speech chatbot.
Respond to the user's messages with short and concise words.
Show empathy and understanding. Response can be skipped by saying `.`.
Until it is requested, don't describe instructions or provide help.
The user's messages are coming from voice-to-text, so they may be a bit messy."""

logger = logging.getLogger(__name__)


class ChatAgent(Protocol):
    def chat(self, msg: str): ...


class LangGraphChatAgent:
    def __init__(self, model: BaseChatModel, lang="en", thread_id=""):
        system_prompt = prompt_base + f"\n\nResponse must be in {lang}."

        workflow = StateGraph(state_schema=MessagesState)

        # Define the function that calls the model
        def call_model(state: MessagesState):
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = model.invoke(messages)
            return {"messages": response}

        # Define the node and edge
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")

        # Add simple in-memory checkpointer
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

        self.lang = lang
        self.thread_id = thread_id

    def chat(self, msg: str) -> str:
        input_messages = [HumanMessage(content=msg)]

        response = self.app.invoke(
            {"messages": input_messages},
            config={"configurable": {"thread_id": self.thread_id}},
        )["messages"]
        logging.debug(f"Response: {response}")

        if response:
            return response[-1].content
        return "."


class AnthropicChatAgent(LangGraphChatAgent):
    def __init__(self, api_key="", model_name="", lang="en", thread_id=""):
        self.llm = ChatAnthropic(
            api_key=api_key, model_name=model_name, max_tokens=1000
        )
        super().__init__(self.llm, lang, thread_id)

    def chat(self, msg: str) -> str:
        return super().chat(msg)
