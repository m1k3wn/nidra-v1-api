from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Define the model names
MODEL_1_NAME = "nidra-v1"
MODEL_2_NAME = "nidra-v2"

# Define the shared prefix
PREFIX = "Interpret this dream: "

# Load both models and their tokenizers
print("Loading nidra-v1...")
tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1_NAME)
model_1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_1_NAME)

print("Loading nidra-v2...")
tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2_NAME)
model_2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_2_NAME)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse request JSON
        data = request.get_json()
        inputs = data.get("inputs", "")
        params = data.get("parameters", {})  # Optional parameters for model.generate()
        model_choice = data.get("model", "nidra-v1")  # Default to nidra-v1

        # Prepend the shared prefix to the input
        full_input = PREFIX + inputs

        # Select the model and tokenizer based on the 'model' parameter
        if model_choice == "nidra-v2":
            tokenizer = tokenizer_2
            model = model_2
        else:
            tokenizer = tokenizer_1
            model = model_1

        # Tokenize input text
        input_ids = tokenizer(full_input, return_tensors="pt").input_ids

        # Generate predictions
        outputs = model.generate(input_ids, **params)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the response
        return jsonify({"generated_text": decoded})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
