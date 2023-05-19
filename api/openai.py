from os.path import join
import json
from typing import Any, Dict, Optional

import openai
import requests
from PIL import Image

from chisel.api.base_api_provider import (
    APIResult, BaseAPIProvider
)
from chisel.model_type import ModelType
from chisel.util.files import get_ext, is_img
from chisel.util.env_handler import EnvHandler


class OpenAI(BaseAPIProvider):
    api_key_name: str = "OPENAI_API_KEY"

    def __init__(self) -> None:
        pass


class OpenAITxtToImg(OpenAI):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = openai.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']


class OpenAIImgToImg(OpenAI):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = openai.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']


class OpenAIImgEdit(OpenAI):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = openai.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']


class OpenAISuperRes(OpenAI):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = openai.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
