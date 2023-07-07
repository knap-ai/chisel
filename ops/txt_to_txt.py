import asyncio
import os
import threading
import queue
from typing import Any, Callable, Optional

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
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

    def __call__(
        self,
        txt: Text,
        callback: Optional[Callable] = None,
    ) -> Text:
        # TODO: generalize. This isn't going to scale.
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


class CustomCallback(BaseCallbackHandler):
    def set_callback(self, callback):
        self.callback = callback

    def on_llm_new_token(self, token: str, **kwargs):
        self.callback.on_llm_new_token(token, kwargs)

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        self.callback.on_llm_end(response=response, kwargs=kwargs)


def openai_chat_thread(g: ThreadedGenerator, prompt, callback):
    custom_callback = CustomCallback()
    custom_callback.set_callback(callback)
    try:
        chat = ChatOpenAI(
            verbose=True,
            streaming=True,
            callbacks=[custom_callback],
            temperature=0.7,
        )
        resp = chat([HumanMessage(content=prompt)])
        return resp
    finally:
        g.close()


def openai_chat(prompt, callback) -> ThreadedGenerator:
    generator = ThreadedGenerator()
    # threading.Thread(target=openai_chat_thread, args=(generator, prompt)).start()
    resp = openai_chat_thread(generator, prompt, callback)
    return resp


def cohere_thread(prompt, callback):
    custom_callback = CustomCallback()
    custom_callback.set_callback(callback)

    llm = Cohere(cohere_api_key=os.environ.get("COHERE_API_KEY"), temperature=0.7)

    template = """{prompt}"""
    prompt_template = PromptTemplate(template=template, input_variables=["prompt"])
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = llm_chain(prompt, callbacks=[custom_callback])


def cohere_llm(prompt: Text, callback) -> None:
    threading.Thread(target=cohere_thread, args=(prompt, callback)).start()
    return None
