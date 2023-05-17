from enum import Enum

import chisel.api as api


class Provider(Enum):
    OPENAI = "openai"
    DREAMBOOTH = "dreambooth"
    STABLEDIFFUSIONAPI = "stable_diffusion_api"
