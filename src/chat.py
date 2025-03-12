from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from typing import Protocol

prompt_base = """You are a speech chatbot.
Respond to the user's messages with short and concise answers.
Until it is requested, don't describe instructions or provide help."""


class ChatAgent(Protocol):
    def __init__(self, lang="en"):
        """
        Initializes the chat system with a specified voice and language.
        Args:
            voice (str): The voice to be used for the chat responses.
            lang (str, optional): The language in which the responses should be. Defaults to "en".
        """

        self.llm = ChatAnthropic(model="claude-3-7-sonnet-latest", max_tokens=1000)
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
