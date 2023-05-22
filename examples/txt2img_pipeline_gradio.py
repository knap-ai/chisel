import gradio as gr

from chisel.ops import ImgToImg, TxtToImg
from chisel.ops.provider import Provider


chisel_result = None


def run_txt2img(txt_input, provider):
    txt2img = TxtToImg(provider=provider)
    output = txt2img(txt_input)
    global chisel_result
    chisel_result = output
    return output.get_image(0)


def run_img2img(txt_input, img_input, provider):
    img2img = ImgToImg(provider=provider)
    if provider == Provider.STABLE_DIFFUSION_API:
        img_input = chisel_result.results[0].get("remote_url", None)
        inputs = [txt_input, img_input]
    elif provider == Provider.STABILITY_AI:
        img_input = chisel_result.results[0].get("img", None)
        inputs = [txt_input, img_input]
    elif provider == Provider.OPENAI:
        img_input = chisel_result.results[0].get("local_filename", None)
        inputs = img_input
    output = img2img(inputs)
    return output.get_image(0)


if __name__ == "__main__":
    radio_choices = [Provider.OPENAI, Provider.STABILITY_AI, Provider.STABLE_DIFFUSION_API]
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
                img_output = gr.Image(shape=(512, 512))

        btn.click(fn=run_txt2img, inputs=[txt_input, provider], outputs=img_output)

        with gr.Row():
            with gr.Column():
                provider = gr.Radio(
                    choices=radio_choices,
                    value=radio_choices[0]
                )
            with gr.Column():
                modify_txt_input = gr.Textbox(value="make the cat cuter and more realistic.")
                img2img_btn = gr.Button("Run")
            with gr.Column():
                img2img_output = gr.Image(shape=(512, 512))

        img2img_btn.click(
            fn=run_img2img,
            inputs=[modify_txt_input, img_output, provider],
            outputs=img2img_output
        )

    demo.launch()
