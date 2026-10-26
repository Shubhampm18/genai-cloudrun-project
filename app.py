import os
import time

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7

app = Flask(__name__)

tokenizer = None
model = None
model_load_seconds = None


def load_model():
    global tokenizer, model, model_load_seconds

    if model is not None:
        return

    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    model.eval()

    model_load_seconds = round(time.perf_counter() - start, 3)


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_ID,
        "device": "cpu",
        "model_loaded": model is not None,
    })


@app.get("/warmup")
def warmup():
    load_model()
    return jsonify({
        "status": "ready",
        "model": MODEL_ID,
        "model_load_seconds": model_load_seconds,
    })


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "Provide a non-empty prompt."}), 400

    if len(prompt) > 4000:
        return jsonify({"error": "Prompt must be 4000 characters or fewer."}), 400

    load_model()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = round(time.perf_counter() - start, 3)
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return jsonify({
        "model": MODEL_ID,
        "response": answer,
        "response_time_seconds": elapsed,
        "output_tokens": int(generated_tokens.shape[0]),
        "model_load_seconds": model_load_seconds,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
