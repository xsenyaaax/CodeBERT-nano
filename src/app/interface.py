# gradio interface

import gradio as gr
from utils.inference import classify_code_no_attention

with gr.Blocks() as demo:
    gr.Markdown("# Code Language Classifier")

    with gr.Row():
        code_input = gr.Textbox(
            label="Paste code here", lines=10, placeholder="Paste your code snippet..."
        )

    output_img = gr.Image(label="Predicted Language", type="filepath")
    output_plot = gr.BarPlot(
        label="Language Probabilities", x="Language", y="Probability"
    )

    classify_button = gr.Button("Classify")
    classify_and_visualize = gr.Button(
        "Classify and Visualize", elem_id="classify_and_viz_btn"
    )

    # unfortunately I could not just use gr.HTML because it does not allow js code, which bertviz uses for attention visualization
    # maybe there is a easier option, but i coult not find it :(
    classify_and_visualize.click(
        None,
        None,
        None,
        js="""
    () => {
        const codeBox = document.querySelector('textarea');
        const fileInput = document.querySelector('input[type="file"]');
        const code = codeBox?.value.trim();
        const hasFile = fileInput?.files?.length > 0;

        if ((code && code.length > 0) || hasFile) {
            const url = '/attention-viz?code=' + encodeURIComponent(code) + '&max_tokens=30';
            window.location.href = url;
        } else {
            alert("Please enter some code.");
        }
    }
""",
    )

    classify_button.click(
        classify_code_no_attention,
        inputs=[code_input],
        outputs=[output_img, output_plot],
    )
