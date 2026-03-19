from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

API_BASE_URL = os.getenv("API_URL", "http://api:8000")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/batch")
def batch():
    return render_template("batch.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=30)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Failed to connect to the backend API container. Is it running?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/explain", methods=["POST"])
def explain():
    """Proxies KernelSHAP explanation — longer timeout needed (60s+)."""
    data = request.json
    try:
        response = requests.post(f"{API_BASE_URL}/explain", json=data, timeout=180)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Failed to connect to the backend API container."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "SHAP explanation timed out. Try a shorter article."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    try:
        files    = {'file': (file.filename, file.stream, file.mimetype)}
        response = requests.post(f"{API_BASE_URL}/predict/csv", files=files, timeout=300)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Failed to connect to the backend API container."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
