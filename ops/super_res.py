from chisel.api import StabilityAISuperRes, StableDiffusionAPISuperRes
from chisel.api.base_api_provider import BaseAPIProvider
from chisel.data_types import Image
from chisel.ops.base_chisel import BaseChisel
from chisel.ops.provider import Provider


class SuperResolution(BaseChisel):
    def __init__(self, provider: Provider) -> None:
        super().__init__(provider)

    def _get_api(self, provider: Provider) -> BaseAPIProvider:
        if provider == Provider.OPENAI:
            return OpenAISuperRes()
        if provider == Provider.STABILITY_AI:
            return StabilityAISuperRes()
        if provider == Provider.STABLE_DIFFUSION_API:
            return StableDiffusionAPISuperRes()

    def __call__(self, img: Image) -> Image:
        return self.api.run(img)
