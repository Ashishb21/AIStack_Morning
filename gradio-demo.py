import gradio as gr
import requests
import json
import os

# Prefer environment variable for security
#POD_ID = os.getenv("RUNPOD_POD_ID", "bwq6ilqz02m6la")
#URL = f"https://{POD_ID}-11434.proxy.runpod.net/api/generate"
URL=f"http://localhost:11434/api/generate"

def generate_text(prompt, model):
    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        with requests.post(URL, json=payload, stream=True, timeout=None) as response:
            response.raise_for_status()
            output = ""

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    output += token
                    yield output

    except Exception as e:
        yield f"❌ Error: {str(e)}"


with gr.Blocks(title="RunPod LLM UI") as demo:
    gr.Markdown("## 🤖 RunPod LLM Playground")

    model_dropdown = gr.Dropdown(
        choices=[
            "gpt-oss-safeguard:20b",
            "llama3:8b",
            "llama3.2"
        ],
        value="llama3:8b",
        label="Select Model"
    )

    prompt_box = gr.Textbox(
        label="Prompt",
        placeholder="Enter your prompt here...",
        lines=4
    )

    output_box = gr.Textbox(
        label="Model Output",
        lines=12
    )

    generate_btn = gr.Button("Generate")

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_box, model_dropdown],
        outputs=output_box
    )

demo.launch()
