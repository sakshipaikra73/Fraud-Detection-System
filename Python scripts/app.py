
import os
import sys
import logging
import pickle
import json
from flask import Flask, request, jsonify
from functools import wraps
from dotenv import load_dotenv
from PIL import Image
import io

# Add parent directory to path to import helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.db.handler import DBHandler
from helpers.risk_engine import RiskEngine
from Notebook.fraud_detection_model_ import extract_features_standalone, predict_url

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Config
API_KEY = os.getenv("API_KEY", "default-insecure-key")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Notebook', 'trained_models', 'best_model.pkl')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Helpers
db = DBHandler()
risk_engine = RiskEngine()

# Load Model
model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Predictions will be limited.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# Security Decorator
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-KEY')
        if key and key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized"}), 401
    return decorated

# Routes

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200

@app.route('/analyze-url', methods=['POST'])
@require_api_key
def analyze_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing URL"}), 400
    
    url = data['url']
    
    # Validation
    if not url or len(url) > 2048:
        return jsonify({"error": "Invalid URL length"}), 400

    try:
        # ML Prediction
        ml_prob = 0.0
        ml_label = False
        if model:
            # Model expects iterable
            ml_prob = model.predict_proba([url])[0][1]
            ml_label = bool(model.predict([url])[0])
            
        # Risk Scoring
        final_score, risk_factors = risk_engine.calculate_risk_score(url, ml_prob)
        risk_level = risk_engine.determine_risk_level(final_score)
        
        # Saving to DB
        db.insert_scan_result(
            url=url,
            is_fraud=final_score > 50, # Threshold
            confidence=final_score, # Using combined score as confidence
            risk_level=risk_level,
            ip_address=request.remote_addr,
            risk_factors=json.dumps(risk_factors)
        )
        
        response = {
            "url": url,
            "is_fraud": final_score > 50, # Threshold
            "confidence": round(final_score, 2),
            "risk_level": risk_level,
            "risk_factors": risk_factors
        }
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/analyze-qr', methods=['POST'])
@require_api_key
def analyze_qr():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        image = Image.open(file.stream)
        from pyzbar.pyzbar import decode
        decoded_objects = decode(image)
        
        if not decoded_objects:
             return jsonify({"error": "No QR code found"}), 400
             
        url = decoded_objects[0].data.decode("utf-8")
        
        # Reuse analysis logic (internal redirect or func call)
        # We'll just call the logic directly to avoid recursion overhead or http overhead
        # Duplicate logic check:
        # ML Prediction
        ml_prob = 0.0
        if model:
             ml_prob = model.predict_proba([url])[0][1]
        
        final_score, _ = risk_engine.calculate_risk_score(url, ml_prob)
        risk_level = risk_engine.determine_risk_level(final_score)
        
        db.insert_scan_result(
            url=url,
            is_fraud=final_score > 50,
            confidence=final_score,
            risk_level=risk_level,
            ip_address=request.remote_addr,
            risk_factors="QR Scan"
        )
        
        return jsonify({
            "url": url,
            "is_fraud": final_score > 50,
            "confidence": round(final_score, 2),
            "risk_level": risk_level
        }), 200
        
    except ImportError:
         return jsonify({"error": "pyzbar not installed on server"}), 500
    except Exception as e:
        logger.error(f"Error processing QR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
@require_api_key
def get_history():
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    try:
        history = db.fetch_history(limit, offset)
        return jsonify(history), 200
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({"error": "Database error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
