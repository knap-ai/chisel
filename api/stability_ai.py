import io
from os import environ
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy
import PIL
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import warnings
from stability_sdk import client as stability_client

from chisel.api.base_api_provider import BaseAPIProvider, APIResult
from chisel.data_types import Image


class StabilityAI(BaseAPIProvider):
    api_key_name: str = "CHISEL_API_KEY_STABILITY_AI"
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
        generation.SAMPLER_DDIM,
        generation.SAMPLER_DDPM,
        generation.SAMPLER_K_EULER,
        generation.SAMPLER_K_EULER_ANCESTRAL,
        generation.SAMPLER_K_HEUN,
        generation.SAMPLER_K_DPM_2,
        generation.SAMPLER_K_DPM_2_ANCESTRAL,
        generation.SAMPLER_K_DPMPP_2S_ANCESTRAL,
        generation.SAMPLER_K_LMS,
        generation.SAMPLER_K_DPMPP_SDE,
        generation.SAMPLER_K_DPMPP_2M,
    ]

    def __init__(self) -> None:
        super().__init__()
        environ["STABILITY_HOST"] = "grpc.stability.ai:443"
        self.api_key = environ.get(self.api_key_name, None)
        if self.api_key is None:
            raise ValueError(
                f"{self.api_key_name} not set. Please set the env variable "
                + "before using this class"
            )
        self.stability_api = stability_client.StabilityInference(
            key=self.api_key,
            verbose=True,
            engine=self.model_engines[0],
        )

    def _process_results(self, results) -> List[Image]:
        # Set up StabilityAPI warning to print to the console if the adult content
        # classifier is tripped. If adult content classifier is not tripped,
        # save generated images.
        api_result = APIResult()

        for i, resp in enumerate(results):
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        f"For result {i}, your request activated the API's safety filters "
                        + "and couldn't be processed. Please modify the prompt and try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = PIL.Image.open(io.BytesIO(artifact.binary))
                    local_filename = self._write_img(
                        img, filename=str(artifact.seed) + ".png"
                    )
                    api_result.add(local_filename, remote_url=None)
        return api_result


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
            "sampler": self.samplers[-1],
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
            "sampler": self.samplers[-1],
        }

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        results = self.stability_api.generate(
            prompt=inp[0],
            init_image=inp[1],
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
            "sampler": self.samplers[-1],
        }

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        results = self.stability_api.generate(
            prompt=inp[0],
            init_image=inp[1],
            mask_image=inp[2],
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

    def __init__(self, upscale_engine: str = None) -> None:
        super().__init__()
        if upscale_engine is None:
            self.engine = self.upscale_engines[-1]
        elif upscale_engine not in self.upscale_engines:
            raise ValueError(
                f"{upscale_engine} is not valid for param " + "'upscale_engine'"
            )
        else:
            self.engine = upscale_engine

        self.stability_upscale_api = stability_client.StabilityInference(
            key=self.api_key,
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
            prompt = self.params.get("prompt", None)
            if inp[0] is not None:
                prompt = inp[0]

            init_image = inp[1]
            if isinstance(inp[1], numpy.ndarray):
                init_image = PIL.Image.fromarray(inp[1])
            elif isinstance(inp[1], str) or isinstance(inp[1], Path):
                init_image = PIL.Image.open(inp[1])

            results = self.stability_upscale_api.upscale(
                init_image=init_image,
                width=self.params["width"],
                prompt=prompt,
                steps=self.params["steps"],
                cfg_scale=self.params["cfg_scale"],
            )
        else:
            results = self.stability_upscale_api.upscale(
                init_image=inp[1],
                width=self.params["width"],
            )
        return self._process_results(results)
