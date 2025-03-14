from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate

ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet-latest"
prompt_base = """You are a speech chatbot.
Respond to the user's messages with short and concise words.
Show empathy and understanding. Response can be skipped by saying `.`.
Until it is requested, don't describe instructions or provide help.
The user's messages are coming from voice-to-text, so they may be a bit messy."""


class ChatAgent(Protocol):
    def chat(self, msg: str): ...


class ChatAnthropicAgent:
    def __init__(self, api_key="", model="", lang="en"):
        """
        Initializes the chat system with a specified voice and language.
        Args:
            voice (str): The voice to be used for the chat responses.
            lang (str, optional): The language in which the responses should be. Defaults to "en".
        """

        self.llm = ChatAnthropic(model=model, max_tokens=1000, api_key=api_key)
        prompt = prompt_base + f"\n\nResponse must be in {lang}."
        self.messages = [
            SystemMessage(content=prompt),
        ]
        self.lang = lang

    async def chat(self, msg: str):
        self.messages.append(HumanMessage(content=msg))
        prompt = ChatPromptTemplate.from_messages(
            self.messages,
        )

        # LCEL
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({"messages": self.messages})
        # print(f"Response: {response}")

        self.messages.append(AIMessage(content=response))
        return response
