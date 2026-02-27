from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

API_URL = os.getenv("API_URL", "http://localhost:8000")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/batch")
def batch():
    return render_template("batch.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data     = request.json
        response = requests.post(f"{API_URL}/predict", json=data)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file     = request.files["file"]
        files    = {"file": (file.filename, file.stream, "text/csv")}
        response = requests.post(f"{API_URL}/predict/csv", files=files)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
