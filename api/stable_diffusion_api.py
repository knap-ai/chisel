from os.path import join
import json
from typing import Any, Dict, Optional

import requests
from PIL import Image

from chisel.api.base_api_provider import (
    ApiResult, BaseAPIProvider
)
from chisel.model_type import ModelType
from chisel.util.files import get_ext, is_img
from chisel.util.env_handler import EnvHandler


class StableDiffusionAPI(BaseAPIProvider):
    api_key_name: str = "SD_API_KEY"
    version: str = "v3"
    base_url: str = "https://stablediffusionapi.com/api/v3"

    def __init__(self):
        super().__init__()
        if not EnvHandler.contains(self.api_key_name):
            raise Exception(f"{self.api_key_name} not set. Please set the env variable " +
                            "before using this class")
        self._key = EnvHandler.get(self.api_key_name)
        self.params = {}

    def get_api_endpoint(self) -> str:
        url = self.base_url
        if self.model_type == ModelType.TXT2IMG:
            url = join(url, "text2img")
        elif self.model_type == ModelType.IMG2IMG:
            url = join(url, "img2img")
        elif self.model_type == ModelType.IMG_EDIT:
            url = join(url, "inpaint")
        elif self.model_type == ModelType.SUPER_RES:
            url = join(url, "super_resolution")
        return url

    def _process_response(self, response: Dict[str, Any]) -> APIResult:
        response_output_list = response.get("output", [])
        api_result = APIResult()
        if isinstance(response_output_list, list):
            for output_url in response_output_list:
                if is_img(output_url):
                    local_filename = self._download_img_from_url(
                        output_url, ext=get_ext(output_url)
                    )
                    api_result.add(
                        local_filename,
                        output_url,
                    )
        else:
            raise ValueError(f"Unexpected value of `response[\"output\"]`: {response_output_list}")

        return api_result


class StableDiffusionAPITxtToImg(StableDiffusionAPI):
    params: Dict[str, str] = {
        "key": "",
        "prompt": "",
        "negative_prompt": None,
        "width": "512",
        "height": "512",
        "samples": "1",
        "num_inference_steps": "20",
        "seed": None,
        "guidance_scale": 7.5,
        "safety_checker": "yes",
        "multi_lingual": "no",
        "panorama": "no",
        "self_attention": "no",
        "upscale": "no",
        "embeddings_model": "embeddings_model_id",
        "webhook": None,
        "track_id": None
    }

    def __init__(self):
        self.model_type = ModelType.TXT2IMG

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        """
        " Run a StableDiffusion API job.
        """
        if params:
            self.set_params(params)

        self.params["key"] = f"{self._key}"

        if isinstance(inp, str):
            self.params["prompt"] = inp
        elif isinstance(inp, list):
            self.params["prompt"] = inp[0]
            self.params["init_image"] = inp[1]

        if isinstance(inp, list):
            for idx, i in enumerate(inp):
                type_i = type(i)
                print(f"elem {idx}: {type_i}")

        headers = {
          'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", self.get_api_endpoint(), headers=headers, data=json.dumps(self.params)
        )
        response_json = response.json()
        return self._process_response(response_json)


class StableDiffusionAPIImgToImg(StableDiffusionAPI):
    params: Dict[str, str] = {
        "key": "",
        "prompt": "",
        "negative_prompt": None,
        "init_image": "",
        "width": "512",
        "height": "512",
        "samples": "1",
        "num_inference_steps": "30",
        "safety_checker": "no",
        "enhance_prompt": "yes",
        "guidance_scale": 7.5,
        "strength": 0.7,
        "seed": None,
        "webhook": None,
        "track_id": None
    }

    def __init__(self):
        self.model_type = ModelType.IMG2IMG
        super().__init__()

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        """
        " Run a StableDiffusion API job.
        """
        if params:
            self.set_params(params)

        self.params["key"] = f"{self._key}"

        if isinstance(inp, str):
            self.params["prompt"] = inp
        elif isinstance(inp, list):
            self.params["prompt"] = inp[0]
            self.params["init_image"] = inp[1]

        if isinstance(inp, list):
            for idx, i in enumerate(inp):
                type_i = type(i)
                print(f"elem {idx}: {type_i}")

        headers = {
          'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", self.get_api_endpoint(), headers=headers, data=json.dumps(self.params)
        )
        response_json = response.json()
        return self._process_response(response_json)


class StableDiffusionAPIImgEdit(StableDiffusionAPI):
    inpainting_params: Dict[str, str] = {
        "negative_prompt": None,
        "init_image": "",
        "mask_image": "",
        "width": "512",
        "height": "512",
        "samples": "1",
        "num_inference_steps": "30",
        "safety_checker": "no",
        "enhance_prompt": "yes",
        "guidance_scale": 7.5,
        "strength": 0.7,
        "seed": None,
        "webhook": None,
        "track_id": None
    }

    def __init__(self):
        self.model_type = ModelType.IMG_EDIT
        super().__init__()

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        """
        " Run a StableDiffusion API job.
        """
        if params:
            self.set_params(params)

        self.params["key"] = f"{self._key}"

        if isinstance(inp, str):
            self.params["prompt"] = inp
        elif isinstance(inp, list):
            self.params["prompt"] = inp[0]
            self.params["init_image"] = inp[1]

        if isinstance(inp, list):
            for idx, i in enumerate(inp):
                type_i = type(i)
                print(f"elem {idx}: {type_i}")

        headers = {
          'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", self.get_api_endpoint(), headers=headers, data=json.dumps(self.params)
        )
        response_json = response.json()
        return self._process_response(response_json)


class StableDiffusionAPISuperRes(StableDiffusionAPI):
    params: Dict[str, str] = {
        "key": "",
        "url": "",
        "scale": 3,
        "webhook": None,
        "face_enhance": False
    }

    def __init__(self):
        self.model_type = ModelType.SUPER_RES
        super().__init__()

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        """
        " Run a StableDiffusion API job.
        """
        if params:
            self.set_params(params)

        self.params["key"] = f"{self._key}"

        if isinstance(inp, str):
            self.params["prompt"] = inp
        elif isinstance(inp, list):
            self.params["prompt"] = inp[0]
            self.params["init_image"] = inp[1]

        if isinstance(inp, list):
            for idx, i in enumerate(inp):
                type_i = type(i)
                print(f"elem {idx}: {type_i}")

        headers = {
          'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", self.get_api_endpoint(), headers=headers, data=json.dumps(self.params)
        )
        response_json = response.json()
        return self._process_response(response_json)
