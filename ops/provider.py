from enum import Enum

import chisel.api as api


class Provider(str, Enum):
    OPENAI = "openai"
    STABILITY_AI = "stability_ai"
    STABLE_DIFFUSION_API = "stable_diffusion_api"
    COHERE = "cohere"
