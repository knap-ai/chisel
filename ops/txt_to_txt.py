import asyncio
import os
import threading
import queue
from typing import Any, Callable, List, Optional

from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Cohere
from langchain.llms.base import LLM
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.vectorstores import Qdrant
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.http import models

from chisel.api.base_api_provider import BaseAPIProvider
from chisel.data_types import Text, Image
from chisel.ops.base_chisel import BaseChisel
from chisel.ops.provider import Provider
from chisel.qdrant_db import QdrantDBFactory


class TxtToTxt(LLM):
    provider: str = Field(default='openai')
    model: str = Field(default='gpt-3.5-turbo')
    use_thread: bool = Field(default=False)
    doc_store: Qdrant = Field(default=None)
    qdrant_client: QdrantClient = Field(default=None)
    embeddings: str = Field(default=OpenAIEmbeddings())
    collection: str = Field(default=None)
    text: str = Field(default="")

    def __init__(
        self,
        provider: Provider,
        model: str,
        use_thread: bool = True,
    ) -> None:
        super().__init__()
        self.use_thread = use_thread
        self.collection = None
        self.doc_store = None
        db_factory = QdrantDBFactory()
        self.qdrant_client = db_factory.get_client()

    def _get_api(
        self,
        provider: Provider,
    ) -> BaseAPIProvider:
        pass

    def build_doc_store(self, collection: str):
        if collection is not None and self.collection != collection:
            self.collection = collection
            qdrant_collections = self.qdrant_client.get_collections()
            if self.collection not in qdrant_collections:
                self.qdrant_client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
                )
            self.doc_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection,
                embeddings=self.embeddings,
            )

    def __call__(
        self,
        txt: Text,
        callback: Optional[Callable] = None,
    ) -> Text:
        # TODO: generalize. This isn't going to scale.
        if self.provider == "openai":
            return openai_chat(
                txt,
                callback,
                self.model,
                self.use_thread,
                self.doc_store,
            )
        elif self.provider == "cohere":
            return cohere_llm(txt, callback, self.use_thread)

    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager = None  # : Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # Used so that TxtToTxt can be passed to Langchain as an LLM.
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        self.text = prompt
        return self.__call__(prompt, run_manager)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager = None  # : Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        final_query = prompt
        self.text = prompt
        if isinstance(prompt, str):
            final_query = [HumanMessage(content=prompt)]

        chat = ChatOpenAI(
            model_name=self.model,
            verbose=True,
            streaming=False,
            temperature=0,
        )
        if self.doc_store:
            qa = RetrievalQA.from_chain_type(
                llm=chat,
                retriever=self.doc_store.as_retriever()
            )
            print(f"prompt: {prompt}")
            resp = await qa.arun(prompt)
            print(f"USING RETRIEVAL QA: {resp}")
        else:
            resp = await chat.agenerate([final_query])

        return resp


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


def openai_chat_thread(
    prompt,
    callback,
    model: str,
    doc_store: Optional = None,
):
    custom_callback = CustomCallback()
    custom_callback.set_callback(callback)

    official_model_name = "gpt-3.5-turbo"
    if model == "gpt-4":
        official_model_name = "gpt-4"

    final_query = prompt
    if isinstance(prompt, str):
        final_query = [HumanMessage(content=prompt)]

    try:
        chat = ChatOpenAI(
            model_name=official_model_name,
            verbose=True,
            streaming=False,
            callbacks=[custom_callback],
            temperature=0.7,
        )
        if doc_store:
            qa = RetrievalQA.from_chain_type(llm=chat, retriever=doc_store.as_retriever())
            resp = qa.run(prompt)
        else:
            resp = chat(final_query)

        return resp
    except Exception as e:
        print("ChiselError: ", e)


def openai_chat(
    prompt,
    callback,
    model: str,
    use_thread: bool,
    doc_store: Optional = None
) -> str:
    if use_thread:
        return threading.Thread(
            target=openai_chat_thread,
            args=(prompt, callback, model, doc_store)
        ).start()
    else:
        return openai_chat_thread(prompt, callback, model, doc_store)


def cohere_thread(prompt, callback):
    custom_callback = CustomCallback()
    custom_callback.set_callback(callback)

    llm = Cohere(cohere_api_key=os.environ.get("COHERE_API_KEY"), temperature=0.7)

    template = """{prompt}"""
    prompt_template = PromptTemplate(template=template, input_variables=["prompt"])
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = llm_chain(prompt, callbacks=[custom_callback])


def cohere_llm(prompt: Text, callback, use_thread: bool) -> str:
    if use_thread:
        return threading.Thread(target=cohere_thread, args=(prompt, callback)).start()
    else:
        return cohere_thread(prompt, callback)
