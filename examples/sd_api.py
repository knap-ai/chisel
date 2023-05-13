import gradio as gr
from chisel.api.stable_diffusion_api import StableDiffusionAPI


chisel_result = None


def run_sd_api(txt_input):
    txt2img = StableDiffusionAPI(model_type="txt_to_img")
    output = txt2img.run(txt_input)
    global chisel_result
    chisel_result = output
    return output.get_image(0)


def run_sd_img2img(txt_input, img_input):
    img2img = StableDiffusionAPI(model_type="img_to_img")
    output = img2img.run(
        [txt_input, chisel_result.results[0].get("remote_url", None)]
    )
    return output.get_image(0)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                txt_input = gr.Textbox(value="a cute cat holding his sombrero")
                btn = gr.Button("Run")
            with gr.Column():
                img_output = gr.Image()

        btn.click(fn=run_sd_api, inputs=txt_input, outputs=img_output)

        with gr.Row():
            with gr.Column():
                modify_txt_input = gr.Textbox(value="make the cat cuter and more realistic.")
                img2img_btn = gr.Button("Run")
            with gr.Column():
                img2img_output = gr.Image()

        img2img_btn.click(
            fn=run_sd_img2img,
            inputs=[modify_txt_input, img_output],
            outputs=img2img_output
        )

    demo.launch()
