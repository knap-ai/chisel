import gradio as gr

from chisel.ops import SuperResolution, TxtToImg
from chisel.ops.provider import Provider


chisel_result = None


def run_txt2img(txt_input, provider):
    txt2img = TxtToImg(provider=provider)
    output = txt2img(txt_input)
    global chisel_result
    chisel_result = output
    return output.get_image(0)


def run_super_resolution(txt_input, img_input, provider):
    super_resolution = SuperResolution(provider=provider)
    if provider == Provider.STABILITY_AI:
        output = super_resolution([txt_input, img_input])
    elif provider == Provider.STABLE_DIFFUSION_API:
        output = super_resolution(
            chisel_result.results[0].get("remote_url", None)
        )
    return output.get_image(0)


if __name__ == "__main__":
    radio_choices = [Provider.OPENAI, Provider.STABILITY_AI, Provider.STABLE_DIFFUSION_API]
    super_res_radio_choices = [Provider.STABILITY_AI, Provider.STABLE_DIFFUSION_API]
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                provider = gr.Radio(
                    choices=radio_choices,
                    value=radio_choices[0]
                )
            with gr.Column():
                txt_input = gr.Textbox(value="a cute cat holding his sombrero")
                btn = gr.Button("Run")
            with gr.Column():
                img_output = gr.Image()

        btn.click(fn=run_txt2img, inputs=[txt_input, provider], outputs=img_output)

        with gr.Row():
            with gr.Column():
                super_res_provider = gr.Radio(
                    choices=super_res_radio_choices,
                    value=super_res_radio_choices[0]
                )
            with gr.Column():
                sr_txt_input = gr.Textbox(
                    value="a cute cat holding his sombrero",
                    label="Additional prompt for use with StabilityAI super resolution."
                )
            with gr.Column():
                super_resolution_btn = gr.Button("Run")
            with gr.Column():
                super_resolution_output = gr.Image()

        super_resolution_btn.click(
            fn=run_super_resolution,
            inputs=[sr_txt_input, img_output, super_res_provider],
            outputs=super_resolution_output
        )

    demo.launch()
