from flask import Flask, render_template, request, jsonify
import requests
import os

# Initialize Flask, pointing to the correct frontend folders based on our Dockerfile structure
app = Flask(
    __name__, 
    template_folder="../frontend/templates", 
    static_folder="../frontend/static"
)

# --- THE FIX: Docker Internal Networking ---
# Instead of localhost, we route to the container name 'api' defined in docker-compose.yml
# We use an environment variable with a fallback just in case.
API_BASE_URL = os.getenv("API_URL", "http://api:8000")

@app.route("/")
def index():
    """Serves the main single-article prediction page."""
    return render_template("index.html")

@app.route("/batch")
def batch():
    """Serves the batch CSV prediction page."""
    return render_template("batch.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """Proxies the single article prediction request to the backend API container."""
    data = request.json
    try:
        # Forwarding request to the internal 'api' container
        response = requests.post(f"{API_BASE_URL}/predict", json=data)
        
        # Pass the exact JSON and status code back to the frontend
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Failed to connect to the backend API container. Is it running?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    """Proxies the CSV upload request to the backend API container."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    
    try:
        # Forward the file to the backend API
        files = {'file': (file.filename, file.stream, file.mimetype)}
        response = requests.post(f"{API_BASE_URL}/predict/csv", files=files)
        
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Failed to connect to the backend API container."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Host must be 0.0.0.0 to be accessible outside the Docker container
    app.run(host="0.0.0.0", port=5000, debug=True)