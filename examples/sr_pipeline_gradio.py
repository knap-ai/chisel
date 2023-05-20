import gradio as gr
import chisel.ops


chisel_result = None


def run_txt2img(txt_input, provider):
    txt2img = TxtToImg(provider=provider)
    output = txt2img(txt_input)
    global chisel_result
    chisel_result = output
    return output.get_image(0)


def run_super_resolution(txt_input, img_input):
    super_resolution = SuperResolution(provider=provider)
    output = super_resolution(
        [txt_input, chisel_result.results[0].get("remote_url", None)]
    )
    return output.get_image(0)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                radio_btn = gr.Radio(
                    choices=['OpenAI', 'StabilityAI', 'StableDiffusionAPI'],
                    value='OpenAI'
                )
            with gr.Column():
                txt_input = gr.Textbox(value="a cute cat holding his sombrero")
                btn = gr.Button("Run")
            with gr.Column():
                img_output = gr.Image()

        btn.click(fn=run_txt2img, inputs=txt_input, outputs=img_output)

        with gr.Row():
            with gr.Column():
                modify_txt_input = gr.Textbox(value="make the cat cuter and more realistic.")
                super_resolution_btn = gr.Button("Run")
            with gr.Column():
                super_resolution_output = gr.Image()

        super_resolution_btn.click(
            fn=run_super_resolution,
            inputs=[modify_txt_input, img_output],
            outputs=super_resolution_output
        )

    demo.launch()
