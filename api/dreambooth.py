from os.path import join
import json
from typing import Any, Dict, Optional

import requests
from PIL import Image

from chisel.api.base_api_provider import (
    BaseAPIProvider, APIResult
)
from chisel.chisel import Chisel
from chisel.model_type import ModelType
from chisel.util.files import get_ext, is_img
from chisel.util.env_handler import EnvHandler


class Dreambooth(BaseAPIProvider):
    api_key_name: str = "DREAMBOOTH_API_KEY"
    version: str = "v3"
    base_url: str = ""

    def __init__(self) -> None:
        pass


class DreamboothTxtToImg(Dreambooth):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = dreambooth.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']


class DreamboothImgToImg(Dreambooth):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = dreambooth.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']


class DreamboothEdit(Dreambooth):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = dreambooth.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']


class DreamboothSuperRes(Dreambooth):
    def __init__(self) -> None:
        pass

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        response = dreambooth.Image.create(
            prompt="a white siamese cat",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
