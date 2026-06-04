"""
Gradio chat demo for granite-ml-coder on a free HF CPU Space.

Uses the 4-bit GGUF via llama-cpp-python (small + fast on CPU) instead of the
full fp32 model (which OOMs the free 16 GB tier).
"""
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

GGUF_REPO = "Tekimax/granite-ml-coder-GGUF"
GGUF_FILE = "granite-ml-coder-Q4_K_M.gguf"
SYSTEM = (
    "You are an expert Python machine-learning engineer. Write correct, runnable "
    "code, explain each pipeline step, watch for overfitting and how gradient "
    "descent converges, and recommend the best model or formula for the task."
)

print("Downloading GGUF …")
model_path = hf_hub_download(GGUF_REPO, GGUF_FILE)
print("Loading model …")
llm = Llama(model_path=model_path, n_ctx=4096, n_threads=2, verbose=False)
print("Model ready.")


def respond(message, history, max_new_tokens, temperature):
    messages = [{"role": "system", "content": SYSTEM}]
    messages += [m for m in history if m.get("role") in ("user", "assistant")]
    messages.append({"role": "user", "content": message})
    # Stream tokens as they're generated so the UI responds immediately
    # instead of waiting for the full answer.
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=0.9,
        stream=True,
    )
    partial = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            partial += delta
            # The model emits raw Python (no markdown fence), which makes Gradio
            # render `# comments` as headings. Wrap it in a python code fence so
            # it renders as code — unless the model already fenced it itself.
            if "```" in partial:
                yield partial
            else:
                yield "```python\n" + partial + "\n```"


demo = gr.ChatInterface(
    respond,
    title="🧪 granite-ml-coder",
    description=(
        "A fine-tuned **IBM Granite 3.1 1B** ML coding assistant (4-bit GGUF). "
        "Free CPU tier → answers take a few seconds; it's a 1B model, so verify "
        "its output. "
        "[Model](https://huggingface.co/Tekimax/granite-ml-coder) · "
        "[Workshop repo](https://github.com/TEKIMAX/tekimax-ML-Model-workshop)"
    ),
    additional_inputs=[
        gr.Slider(64, 512, value=320, step=32, label="max new tokens"),
        gr.Slider(0.0, 1.2, value=0.7, step=0.1, label="temperature"),
    ],
    examples=[
        ["Write a scikit-learn pipeline to classify the iris dataset and report accuracy."],
        ["Explain overfitting and how to detect it with a train/validation curve."],
        ["Build a small Keras autoencoder for anomaly detection on tabular data."],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.queue().launch()
