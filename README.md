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
