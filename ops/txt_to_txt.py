import asyncio
import os
import threading
import queue
from typing import Callable, Optional

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Cohere
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from chisel.api.base_api_provider import BaseAPIProvider
from chisel.data_types import Text, Image
from chisel.ops.base_chisel import BaseChisel
from chisel.ops.provider import Provider


class TxtToTxt(BaseChisel):
    def __init__(self, provider: Provider) -> None:
        super().__init__(provider)
        self.provider = provider

    def _get_api(self, provider: Provider) -> BaseAPIProvider:
        pass

    def __call__(self, txt: Text, callback: Optional[Callable] = None) -> Text:
        if self.provider == "openai":
            return openai_chat(txt, callback)
        elif self.provider == "cohere":
            return cohere_llm(txt, callback)


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        print("NEW TOKEN: ", token)
        self.gen.send(token)


async def openai_chat_thread(g: ThreadedGenerator, prompt, callback):
    try:
        chat = ChatOpenAI(
            verbose=True,
            streaming=True,
            # callback_manager=BaseCallbackManager([ChainStreamHandler(g)]),
            temperature=0.7,
        )
        resp = chat([HumanMessage(content=prompt)])
        callback(response=resp)

    finally:
        g.close()


def openai_chat(prompt, callback) -> ThreadedGenerator:
    generator = ThreadedGenerator()
    # threading.Thread(target=openai_chat_thread, args=(generator, prompt)).start()
    openai_chat_thread(generator, prompt, callback)
    return generator


def cohere_thread(prompt, callback):
    llm = Cohere(cohere_api_key=os.environ.get(COHERE_API_KEY), temperature=0.7)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(prompt)
    callback(response=answer)


def cohere_llm(prompt: Text, callback) -> None:
    threading.Thread(target=cohere_thread, args=(prompt, callback)).start()
    return None
