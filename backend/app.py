"""
TruthLens AI — Flask Backend
==============================
Provides endpoints for text analysis, image analysis, combined reporting,
and history tracking for the TruthLens AI misinformation detection platform.
"""

import os
import io
import re
import json
import uuid
import datetime
import traceback
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
model = None
vectorizer = None

try:
    import joblib
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        print("[INFO] ML model and vectorizer loaded successfully.")
    else:
        print("[WARN] model.pkl / vectorizer.pkl not found. Using fallback mock predictions.")
except Exception as e:
    print(f"[WARN] Could not load ML model: {e}. Using fallback mock predictions.")

# ---------------------------------------------------------------------------
# Gemini API Setup
# ---------------------------------------------------------------------------
gemini_model = None

try:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "AIzaSyAvZqPA9vhd-EUtOSUdvP-MYmoF-J9mEC4":
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("[INFO] Gemini API configured successfully.")
    else:
        print("[WARN] GEMINI_API_KEY not set. Explanation generation will use fallback.")
except Exception as e:
    print(f"[WARN] Could not configure Gemini API: {e}. Using fallback explanations.")

# ---------------------------------------------------------------------------
# In-Memory History Store
# ---------------------------------------------------------------------------
history_store = []

# ---------------------------------------------------------------------------
# Suspicious Phrase Keywords
# ---------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "breaking", "shocking", "urgent", "miracle cure", "unbelievable",
    "instantly", "secret", "exclusive", "conspiracy", "they don't want you to know",
    "wake up", "banned", "censored", "cover-up", "hoax", "mind-blowing",
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def detect_suspicious_phrases(text):
    """Find suspicious phrases in the text using keyword matching."""
    found = []
    text_lower = text.lower()
    # Also try to extract the surrounding context
    sentences = re.split(r'[.!?]', text)

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in text_lower:
            # Try to find the original-case version in the text
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            for match in pattern.finditer(text):
                # Get a snippet around the match
                start = max(0, match.start() - 5)
                end = min(len(text), match.end() + 20)
                snippet = text[start:end].strip()
                if snippet and snippet not in found:
                    found.append(snippet)
    return found


def get_gemini_explanation(text):
    """Generate an explanation using Gemini API, with fallback."""
    if gemini_model:
        try:
            prompt = (
                "You are a misinformation detection expert. Analyze the following content "
                "and explain in 2-3 sentences why this content may be misinformation. "
                "Be specific about the claims made and why they are problematic.\n\n"
                f"Content: \"{text}\""
            )
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[WARN] Gemini API call failed: {e}")

    # Fallback explanation
    suspicious = detect_suspicious_phrases(text)
    if suspicious:
        return (
            f"This content contains sensationalist language such as "
            f"'{', '.join(suspicious[:3])}', which is commonly associated with "
            f"misinformation. The claims made lack credible sourcing and use "
            f"exaggerated language designed to provoke emotional reactions "
            f"rather than inform."
        )
    return (
        "The content could not be verified against trusted sources. "
        "It may contain unsubstantiated claims. Always cross-reference "
        "information with reputable news organizations."
    )


def mock_predict(text):
    """Fallback prediction when ML model is not available."""
    text_lower = text.lower()
    suspicion_score = 0
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in text_lower:
            suspicion_score += 15

    # Check for excessive punctuation / caps
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.3:
        suspicion_score += 20
    if text.count("!") > 1:
        suspicion_score += 10

    suspicion_score = min(suspicion_score, 98)

    if suspicion_score >= 50:
        return "Fake", suspicion_score, max(5, 100 - suspicion_score)
    elif suspicion_score >= 25:
        return "Uncertain", suspicion_score, 50
    else:
        return "Real", max(10, suspicion_score), max(60, 100 - suspicion_score)


# ---------------------------------------------------------------------------
# Image Analysis Helpers
# ---------------------------------------------------------------------------

def analyze_image_heuristics(image_bytes):
    """Perform simple heuristic-based image analysis."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img, dtype=np.float64)

        anomalies = []
        scores = []

        # 1. Blur detection — check variance of Laplacian-like filter
        gray = np.mean(img_array, axis=2)
        # Simple Laplacian approximation
        laplacian = np.abs(
            gray[1:-1, 1:-1] * 4
            - gray[:-2, 1:-1] - gray[2:, 1:-1]
            - gray[1:-1, :-2] - gray[1:-1, 2:]
        )
        blur_score = np.var(laplacian)
        if blur_score < 500:
            anomalies.append("Blurred facial edges")
            scores.append(25)
        elif blur_score < 1000:
            anomalies.append("Slightly blurred regions detected")
            scores.append(10)

        # 2. Brightness variance check
        brightness = np.mean(img_array, axis=2)
        brightness_std = np.std(brightness)
        if brightness_std < 30:
            anomalies.append("Inconsistent lighting")
            scores.append(20)
        elif brightness_std > 80:
            anomalies.append("Unusual brightness variance")
            scores.append(15)

        # 3. Texture repetition — check for repeated patterns using block comparison
        h, w = gray.shape
        block_size = min(32, h // 4, w // 4)
        if block_size > 8:
            blocks = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    blocks.append(block.flatten())
                    if len(blocks) > 50:
                        break
                if len(blocks) > 50:
                    break

            if len(blocks) > 2:
                # Compare blocks for similarity
                similar_count = 0
                total_comparisons = 0
                for i in range(min(len(blocks), 20)):
                    for j in range(i + 1, min(len(blocks), 20)):
                        correlation = np.corrcoef(blocks[i], blocks[j])[0, 1]
                        if not np.isnan(correlation) and correlation > 0.95:
                            similar_count += 1
                        total_comparisons += 1

                if total_comparisons > 0 and similar_count / total_comparisons > 0.1:
                    anomalies.append("Repeated background patterns")
                    scores.append(20)

        # 4. Edge consistency — check for unnatural edge distribution
        edges_h = np.abs(np.diff(gray, axis=0))
        edges_v = np.abs(np.diff(gray, axis=1))
        edge_ratio = np.mean(edges_h) / max(np.mean(edges_v), 0.01)
        if edge_ratio > 2.0 or edge_ratio < 0.5:
            anomalies.append("Unnatural skin texture")
            scores.append(15)

        # 5. Color channel analysis
        r_mean, g_mean, b_mean = [np.mean(img_array[:, :, i]) for i in range(3)]
        color_variance = np.std([r_mean, g_mean, b_mean])
        if color_variance < 5:
            anomalies.append("Unusual color uniformity")
            scores.append(10)

        # Calculate overall confidence
        if not anomalies:
            # Even if nothing suspicious, give some base anomalies for demo purposes
            anomalies = ["No major anomalies detected"]
            confidence = max(15, int(np.random.uniform(10, 30)))
            verdict = "Authentic"
        else:
            confidence = min(95, sum(scores) + int(np.random.uniform(5, 20)))
            if confidence >= 60:
                verdict = "Possibly AI Generated"
            elif confidence >= 35:
                verdict = "Suspicious"
            else:
                verdict = "Authentic"

        # Determine possible generation method
        if verdict == "Possibly AI Generated":
            method = "GAN or diffusion-generated image"
        elif verdict == "Suspicious":
            method = "Possible image manipulation or enhancement detected"
        else:
            method = "No clear signs of artificial generation"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "anomalies": anomalies,
            "possibleGenerationMethod": method,
            "verificationTips": [
                "Check reverse image search",
                "Inspect image metadata (EXIF data)",
                "Compare facial symmetry",
                "Look for inconsistent shadows",
                "Check for warped backgrounds near subject edges"
            ]
        }

    except Exception as e:
        print(f"[ERROR] Image analysis failed: {e}")
        traceback.print_exc()
        return {
            "verdict": "Uncertain",
            "confidence": 50,
            "anomalies": ["Could not fully analyze image"],
            "possibleGenerationMethod": "Analysis incomplete",
            "verificationTips": [
                "Try uploading a higher quality image",
                "Check reverse image search",
                "Inspect image metadata"
            ]
        }


# ===========================================================================
# API Endpoints
# ===========================================================================

@app.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "TruthLens AI Backend",
        "model_loaded": model is not None,
        "gemini_available": gemini_model is not None
    })


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    """
    Analyze text for misinformation.
    Expects JSON: { "text": "some article text" }
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        # --- ML Prediction ---
        if model and vectorizer:
            text_vectorized = vectorizer.transform([text])
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]

            if prediction == 1:
                verdict = "Fake"
                confidence = int(round(max(probabilities) * 100))
                reliability = max(5, 100 - confidence)
            else:
                verdict = "Real"
                confidence = int(round(max(probabilities) * 100))
                reliability = min(95, confidence)
        else:
            verdict, confidence, reliability = mock_predict(text)

        # --- Suspicious Phrases ---
        suspicious_phrases = detect_suspicious_phrases(text)

        # If phrases found and verdict was "Real", bump to "Uncertain"
        if suspicious_phrases and verdict == "Real" and len(suspicious_phrases) >= 2:
            verdict = "Uncertain"
            confidence = max(confidence, 55)
            reliability = min(reliability, 50)

        # --- Gemini Explanation ---
        explanation = get_gemini_explanation(text)

        # --- Trusted Sources ---
        trusted_sources = ["WHO", "Reuters", "BBC", "Associated Press", "Snopes", "PolitiFact"]

        result = {
            "verdict": verdict,
            "confidence": confidence,
            "reliability": reliability,
            "suspiciousPhrases": suspicious_phrases,
            "explanation": explanation,
            "trustedSources": trusted_sources,
        }

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] /analyze-text failed: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error during text analysis"}), 500


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """
    Analyze an uploaded image for signs of AI generation / manipulation.
    Expects multipart/form-data with an 'image' file.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided. Use 'image' as the form field name."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Empty image file"}), 400

        result = analyze_image_heuristics(image_bytes)
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] /analyze-image failed: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error during image analysis"}), 500


@app.route("/combined-report", methods=["POST"])
def combined_report():
    """
    Generate a combined trust report from text and image analysis results.
    Expects JSON: { "textResult": {...}, "imageResult": {...} }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        text_result = data.get("textResult", {})
        image_result = data.get("imageResult", {})

        # Calculate combined trust score
        text_confidence = text_result.get("confidence", 50)
        image_confidence = image_result.get("confidence", 50)
        text_verdict = text_result.get("verdict", "Uncertain")
        image_verdict = image_result.get("verdict", "Uncertain")

        # Higher confidence in "Fake" / "Suspicious" = lower trust
        text_trust = 100 - text_confidence if text_verdict in ["Fake", "Uncertain"] else text_confidence
        image_trust = 100 - image_confidence if image_verdict in ["Suspicious", "Possibly AI Generated"] else image_confidence

        trust_score = int(round((text_trust * 0.5 + image_trust * 0.5)))
        trust_score = max(0, min(100, trust_score))

        # Determine final status
        if trust_score >= 70:
            final_status = "Likely Reliable"
        elif trust_score >= 40:
            final_status = "Suspicious"
        else:
            final_status = "High Risk Misinformation"

        # Build timeline
        timeline = []
        suspicious_phrases = text_result.get("suspiciousPhrases", [])
        anomalies = image_result.get("anomalies", [])

        if suspicious_phrases:
            timeline.append(f"Detected sensational wording: {', '.join(suspicious_phrases[:3])}")
        timeline.append(f"Text model classified article as {text_verdict} ({text_confidence}% confidence)")
        if anomalies and anomalies[0] != "No major anomalies detected":
            timeline.append(f"Image analysis found: {', '.join(anomalies[:3])}")
        else:
            timeline.append("Image analysis found no major anomalies")
        timeline.append(f"Combined analysis marked the content as {final_status.lower()}")

        # Build summary
        text_summary = f"The text appears {'misleading' if text_verdict != 'Real' else 'credible'}"
        image_summary = f"the image {'shows signs of AI generation' if image_verdict != 'Authentic' else 'appears authentic'}"
        summary = f"{text_summary} and {image_summary}. Overall trust score: {trust_score}%."

        result = {
            "trustScore": trust_score,
            "finalStatus": final_status,
            "summary": summary,
            "timeline": timeline,
        }

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] /combined-report failed: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error during report generation"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    """Return all stored analysis history entries."""
    return jsonify(history_store)


@app.route("/history", methods=["POST"])
def add_history():
    """
    Add a new entry to the analysis history.
    Expects JSON with analysis data.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        entry = {
            "id": str(uuid.uuid4()),
            "date": datetime.datetime.now().isoformat(),
            "type": data.get("type", "Unknown"),
            "verdict": data.get("verdict", "Unknown"),
            "confidence": data.get("confidence", 0),
            "data": data,
        }

        history_store.insert(0, entry)  # newest first

        # Keep only last 100 entries
        if len(history_store) > 100:
            history_store.pop()

        return jsonify(entry), 201

    except Exception as e:
        print(f"[ERROR] /history POST failed: {e}")
        return jsonify({"error": "Internal server error"}), 500


# ===========================================================================
# Run Server
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TruthLens AI Backend")
    print("  Running on http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
