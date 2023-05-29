# Chisel - Compose AI for vision applications.

The **[documentation](https://knap.ai/docs)** are a great place to start building with Chisel.

## Getting Started

Install with pip:

```bash
pip install chisel-ai
```

## Operations

Chisel supports several AI model-powered operations:

1. `TxtToImg`
2. `ImgToImg`
3. `ImgEdit`
4. `SuperResolution`

These operations make it very easy to start building AI vision pipelines:

```python
from chisel.ops import ImgToImg, TxtToImg, SuperResolution
from chisel.ops.provider import Provider

txt2img = TxtToImg(provider=Provider.STABILITY_AI)
img2img = ImgToImg(provider=Provider.OPENAI)
super_res = SuperResolution(provider=Provider.STABILITY_AI)

prompt = "watercolor painting of a park in the fall"

img = txt2img(prompt)
refined_img = img2img(img)
upscaled_img = super_res(refined_img)
```

## Initial Setup

### API Keys

Chisel looks for API keys in these environment variables

- OpenAI - `CHISEL_API_KEY_OPEN_AI`
- StabilityAI - `CHISEL_API_KEY_STABILITY_AI`
- Stable Diffusion - `CHISEL_API_KEY_STABLE_DIFFUSION`

### OpenAI Setup

- [Create an OpenAI API key](https://platform.openai.com/account/api-keys)
- Set Chisel's environment variable
  - ZSH - `echo 'export CHISEL_API_KEY_OPEN_AI=1234ABCD' >> ~/.zshenv`
  - BASH - `echo 'export CHISEL_API_KEY_OPEN_AI_=12345ABCD' >> ~/.bash_profile`

### StabilityAI Setup

- [Create a StabilityAI API Key](https://platform.stability.ai/docs/getting-started/authentication)
- Set Chisel's environment variable
  - ZSH - `echo 'export CHISEL_API_KEY_STABILITY_AI=1234ABCD' >> ~/.zshenv`
  - BASH - `echo 'export CHISEL_API_KEY_STABILITY_AI=12345ABCD' >> ~/.bash_profile`

### Stable Diffusion Setup

- [Create a Stable Diffusion API key](https://stablediffusionapi.com)
- Set Chisel's environment variable
  - ZSH - `echo 'export CHISEL_API_KEY_STABLE_DIFFUSION=1234ABCD' >> ~/.zshenv`
  - BASH - `echo 'export CHISEL_API_KEY_STABLE_DIFFUSION=12345ABCD' >> ~/.bash_profile`
