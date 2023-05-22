from typing import List, Union
from chisel.api import StabilityAISuperRes, StableDiffusionAPISuperRes
from chisel.api.base_api_provider import BaseAPIProvider
from chisel.data_types import Image
from chisel.ops.base_chisel import BaseChisel
from chisel.ops.provider import Provider


class SuperResolution(BaseChisel):
    def __init__(self, provider: Provider) -> None:
        super().__init__(provider)

    def _get_api(self, provider: Provider) -> BaseAPIProvider:
        if provider == Provider.STABILITY_AI:
            return StabilityAISuperRes()
        elif provider == Provider.STABLE_DIFFUSION_API:
            return StableDiffusionAPISuperRes()
        else:
            raise ValueError(f"Invalid provider: {provider}")

    def __call__(self, inputs: Union[Image, List[str]]) -> Image:
        return self.api.run(inputs)
