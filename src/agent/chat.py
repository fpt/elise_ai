import logging
from typing import AsyncGenerator

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
from langgraph.graph import END, START, MessagesState, StateGraph  # type: ignore
from langgraph.prebuilt import ToolNode  # type: ignore

from .tools import get_local_datetime, remind_memory, save_memory

ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet-latest"
prompt_base = """You are a speech chatbot.
Respond to the user's messages with short and concise words.
Show empathy and understanding.
Until it is requested, don't describe instructions or provide help.
The user's messages are coming from voice-to-text, so they may be a bit messy."""
MAX_TOKENS = 10000
logger = logging.getLogger(__name__)


class LangGraphChatAgent:
    def __init__(self, model: BaseChatModel, lang="en", thread_id=""):
        system_prompt = prompt_base + f"\n\nResponse must be in {lang}."

        workflow = StateGraph(state_schema=MessagesState)

        tools = [get_local_datetime, remind_memory]
        model_with_tools = model.bind_tools(tools)

        # Define the function that calls the model
        # def call_model(state: MessagesState):
        #     messages = [SystemMessage(content=system_prompt)] + state["messages"]
        #     response = model.invoke(messages)
        #     return {"messages": response}
        def call_model(state: MessagesState):
            system_message = SystemMessage(content=system_prompt)
            message_history = state["messages"][
                :-1
            ]  # exclude the most recent user input or tool result
            # Summarize the messages if the chat history reaches a certain size
            # Note that last message may be a tool call
            if len(message_history) >= 4 and not message_history[-1].tool_calls:
                last_human_message = state["messages"][-1]
                # Invoke the model to generate conversation summary
                summary_prompt = (
                    "Distill the above chat messages into a single Markdown document. "
                    "There should be a title, summary, related keywords and a list of details. "
                    "Include as many specific details as you can. "
                )
                summary_message = model.invoke(
                    message_history + [HumanMessage(content=summary_prompt)]
                )
                # Save the summary message to a file
                # Handle case where content is not a simple string
                if isinstance(summary_message.content, str):
                    save_memory(summary_message.content)
                else:
                    # For content that is a list of strings/dicts, extract text content and join
                    text_content = ""
                    for item in summary_message.content:
                        if isinstance(item, str):
                            text_content += item
                        elif isinstance(item, dict) and "text" in item:
                            text_content += item["text"]
                    save_memory(text_content)

                summary_message.additional_kwargs["type"] = "summary"

                # Delete messages that we no longer want to show up
                delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
                # Re-add user message
                human_message = HumanMessage(content=last_human_message.content)
                # Call the model with summary & response
                logger.debug(
                    f"Calling model with {system_message}, {summary_message}, {human_message}"
                )
                response = model_with_tools.invoke(
                    [system_message, summary_message, human_message]
                )

                message_updates = [
                    summary_message,
                    human_message,
                    response,
                ] + delete_messages
            else:
                logger.debug(
                    f"Calling model with {system_message} and messages: {state['messages']}"
                )
                message_updates = [
                    model_with_tools.invoke([system_message] + state["messages"])
                ]

            return {"messages": message_updates}

        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END

        tool_node = ToolNode(tools)

        # Define the node and edge
        workflow.add_node("model", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "model")
        workflow.add_conditional_edges("model", should_continue, ["tools", END])
        workflow.add_edge("tools", "model")

        # Add simple in-memory checkpointer
        memory = InMemorySaver()
        self.app = workflow.compile(checkpointer=memory)

        self.lang = lang
        self.thread_id = thread_id

    async def chat(self, msg: str) -> AsyncGenerator[str, None]:
        input_messages = [HumanMessage(content=msg)]

        async for kind, chunk in self.app.astream(
            {"messages": input_messages},
            config={"configurable": {"thread_id": self.thread_id}},
            stream_mode=["updates"],
        ):
            if kind != "updates":
                logger.debug(f"Skipping kind: {kind}")
                continue

            if "model" in chunk:
                responses = chunk["model"]["messages"]
                # logging.info(f"model.messages: {responses}")

                for response in responses:
                    if type(response) is not AIMessage:
                        # Skip non-AI messages
                        continue
                    if response.additional_kwargs.get("type") == "summary":
                        # Skip summary messages
                        continue

                    content = response.content

                    if type(content) is list:
                        # Workaround for the case where the model returns multiple messages
                        # e.g. tool calling may return multiple messages with 'type'='text' and 'type'='tool_use'
                        # find 'type'='text' in response
                        for r in content:
                            # r is either str or dict
                            if isinstance(r, str):
                                yield r
                            elif isinstance(r, dict):
                                # Add type hint for mypy
                                if r["type"] == "text":
                                    yield r["text"]
                                elif r["type"] == "tool_use":
                                    logging.debug(
                                        f"Tool use: {r['name']} id: {r['id']}"
                                    )
                                    pass
                    elif type(content) is str:
                        if content.strip():
                            yield content
                        else:
                            logging.info(f"Empty response: {response}")
                    else:
                        logging.error(f"Unknown content type: {type(content)}")
            elif "tools" in chunk:
                pass  # do nothing
            else:
                logging.error(f"Unknown chunk: {chunk}")


class AnthropicChatAgent(LangGraphChatAgent):
    def __init__(self, api_key="", model_name="", lang="en", thread_id=""):
        self.llm = ChatAnthropic(
            api_key=api_key, model_name=model_name, max_tokens=MAX_TOKENS
        )
        super().__init__(self.llm, lang, thread_id)
        logging.info(f"AnthropicChatAgent initialized with model {model_name}")


class OpenAIChatAgent(LangGraphChatAgent):
    def __init__(self, api_key="", model="", lang="en", thread_id=""):
        self.llm = ChatOpenAI(api_key=api_key, model=model, max_tokens=MAX_TOKENS)
        super().__init__(self.llm, lang, thread_id)
        logging.info(f"OpenAIChatAgent initialized with model {model}")


class OllamaChatAgent(LangGraphChatAgent):
    def __init__(self, host="", port="", model="", lang="en", thread_id=""):
        self.llm = ChatOllama(
            base_url=f"http://{host}:{port}",
            model=model,
            max_tokens=MAX_TOKENS,
        )
        super().__init__(self.llm, lang, thread_id)
        logging.info(f"OllamaChatAgent initialized with model {model}")
