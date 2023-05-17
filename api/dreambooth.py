from os import environ
from os.path import join
import json
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from stability_sdk import stability_client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

from chisel.api.base_api_provider import (
    BaseAPIProvider, APIResult
)
from chisel.chisel import Chisel
from chisel.model_type import ModelType
from chisel.util.files import get_ext, is_img
from chisel.util.env_handler import EnvHandler


class StabilityAI(BaseAPIProvider):
    api_key_name: str = "STABILITY_KEY"
    version: str = "v3"
    base_url: str = ""
    model_engines: List[str] = [
        "stable-diffusion-xl-beta-v2-2-2",
        "stable-diffusion-v1",
        "stable-diffusion-v1-5",
        "stable-diffusion-512-v2-0",
        "stable-diffusion-768-v2-0",
        "stable-diffusion-512-v2-1",
        "stable-diffusion-768-v2-1",
        "stable-diffusion-xl-beta-v2-2-2",
        "stable-inpainting-v1-0",
        "stable-inpainting-512-v2-0",
    ]
    samplers: List[str] = [
        "ddim",
        "plms",
        "k_euler",
        "k_euler_ancestral",
        "k_heun",
        "k_dpm_2",
        "k_dpm_2_ancestral",
        "k_dpmpp_2s_ancestral",
        "k_lms",
        "k_dpmpp_2m",
        "k_dpmpp_sde",
    ]

    def __init__(self) -> None:
        environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        self.stability_api = stability_client.StabilityInference(
            key=environ[self.api_key_name],
            verbose=True,
            engine=self.model_engines[0],
        )

    def _process_results(self, results) -> List[Image]:
        # Set up StabilityAPI warning to print to the console if the adult content
        # classifier is tripped. If adult content classifier is not tripped,
        # save generated images.
        img_results = []
        for i, resp in enumerate(results):
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        f"For result {i}, your request activated the API's safety filters " +
                        "and couldn't be processed. Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = PIL.Image.open(io.BytesIO(artifact.binary))
                    img.save(str(artifact.seed)+ ".png")
                    img_results.append(img)
        return img_results


class StabilityAITxtToImg(StabilityAI):
    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {
            "seed": None,
            "steps": 30,
            "cfg_scale": 8.0,
            "width": 512,
            "height": 512,
            "samples": 1,
            "sampler": "k_dpmpp_2m",
        }

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        self.set_params(params)
        results = self.stability_api.generate(
            prompt=inp,
            seed=self.params["seed"],
            steps=self.params["steps"],
            cfg_scale=self.params["cfg_scale"],
            width=self.params["width"],
            height=self.params["height"],
            samples=self.params["samples"],
            sampler=self.params["sampler"],
        )
        return self._process_results(results)


class StabilityAIImgToImg(StabilityAI):
    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {
            "seed": None,
            "steps": 30,
            "start_schedule": 1,
            "cfg_scale": 8.0,
            "width": 512,
            "height": 512,
            "samples": 1,
            "sampler": "k_dpmpp_2m",
        }

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        results = self.stability_api.generate(
            prompt=inp,
            init_image=img,
            seed=self.params["seed"],
            start_schedule=self.params["start_schedule"],
            steps=self.params["steps"],
            cfg_scale=self.params["cfg_scale"],
            width=self.params["width"],
            height=self.params["height"],
            samples=self.params["samples"],
            sampler=self.params["sampler"],
        )
        return self._process_results(results)


class StabilityAIImgEdit(StabilityAI):
    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {
            "seed": None,
            "start_schedule": 1,
            "steps": 30,
            "cfg_scale": 8.0,
            "width": 512,
            "height": 512,
            "samples": 1,
            "sampler": "k_dpmpp_2m",
        }

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        results = self.stability_api.generate(
            prompt=inp,
            init_image=img,
            mask_image=mask,
            seed=self.params["seed"],
            start_schedule=self.params["start_schedule"],
            steps=self.params["steps"],
            cfg_scale=self.params["cfg_scale"],
            width=self.params["width"],
            height=self.params["height"],
            samples=self.params["samples"],
            sampler=self.params["sampler"],
        )
        return self._process_results(results)


class StabilityAISuperRes(StabilityAI):
    upscale_engines: List[str] = [
        "esrgan-v1-x2plus",
        "stable-diffusion-x4-latent-upscaler",
    ]

    def __init__(self, upscale_engine: str) -> None:
        if upscale_engine not in self.upscale_engines:
            raise ValueError(f"{upscale_engine} is not valid for param " +
                             "'upscale_engine'".)
        else:
            self.engine = upscale_engine

        self.stability_upscale_api = client.StabilityInference(
            key=os.environ[super().api_key_name],
            upscale_engine=self.engine,
            verbose=True,
        )
        self.params: Dict[str, Any] = {
            "width": 1024,
            "prompt": None,
            "seed": None,
            "steps": 30,
            "cfg_scale": 8.0,
        }

    def get_upscale_engines(self) -> List[str]:
        return self.upscale_engines

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        if self.engine == "stable-diffusion-x4-latent-upscaler":
            self.stability_upscale_api.upscale(
                init_image=inp,
                width=self.params["width"],
                prompt=self.params["prompt"],
                seed=self.params["seed"],
                steps=self.params["steps"],
                cfg_scale=self.params["cfg_scale"],
            )
        else:
            self.stability_upscale_api.upscale(
                init_image=inp,
                width=self.params["width"],
            )
        return self._process_results(results)
