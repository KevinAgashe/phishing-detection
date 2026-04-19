import os
import re
from typing import Dict, List, Tuple

import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

PHISHING_KEYWORDS = ["urgent", "verify", "bank", "password", "login", "account", "confirm"]
SUSPICIOUS_SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "shorturl"]
TRUSTED_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "company.com", "edu.org"]


def load_artifacts():
    """Load trained model and vectorizer from disk."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def extract_features(email_text: str, sender_email: str) -> Dict[str, object]:
    """Extract rule-based indicators from email content and sender."""
    lowered_text = email_text.lower()
    matched_keywords = [word for word in PHISHING_KEYWORDS if word in lowered_text]
    links = re.findall(r"(https?://[^\s]+|www\.[^\s]+)", email_text, flags=re.IGNORECASE)

    suspicious_links = []
    for link in links:
        link_lower = link.lower()
        if "http://" in link_lower or any(short in link_lower for short in SUSPICIOUS_SHORTENERS):
            suspicious_links.append(link)

    sender_is_valid = bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", sender_email.strip()))
    sender_domain = sender_email.split("@")[-1].lower().strip() if "@" in sender_email else ""
    domain_mismatch = sender_is_valid and sender_domain not in TRUSTED_DOMAINS

    special_char_count = len(re.findall(r"[@#$%^&*()!~`_=+{}\[\]|\\:;\"'<>,.?/-]", email_text))

    return {
        "matched_keywords": matched_keywords,
        "link_count": len(links),
        "suspicious_links": suspicious_links,
        "sender_is_valid": sender_is_valid,
        "domain_mismatch": domain_mismatch,
        "special_char_count": special_char_count,
    }


def rule_based_detection(features: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Evaluate heuristic phishing rules and return reasons."""
    reasons = []
    is_phishing = False

    if features["matched_keywords"]:
        reasons.append(f"Contains suspicious keywords: {', '.join(features['matched_keywords'])}")
        is_phishing = True

    if features["link_count"] > 2:
        reasons.append(f"Contains many links ({features['link_count']}).")
        is_phishing = True

    if features["suspicious_links"]:
        reasons.append("Contains suspicious URL pattern(s): " + ", ".join(features["suspicious_links"]))
        is_phishing = True

    if not features["sender_is_valid"]:
        reasons.append("Sender email format appears invalid.")
        is_phishing = True
    elif features["domain_mismatch"]:
        reasons.append("Sender domain looks unusual or untrusted.")
        is_phishing = True

    if features["special_char_count"] > 12:
        reasons.append("Unusually high number of special characters.")
        is_phishing = True

    return is_phishing, reasons




def load_ann_model():
    model = load_model("ann_model.h5")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

def model_prediction(email_text):
    model, vectorizer = load_ann_model()
    transformed = vectorizer.transform([email_text]).toarray()

    prediction = model.predict(transformed)[0][0]
    
    # Convert to 0 or 1
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = float(prediction if predicted_class == 1 else 1 - prediction)

    return predicted_class, confidence


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email_content", "").strip()
    sender_email = request.form.get("sender_email", "").strip()

    if not email_text or not sender_email:
        return render_template(
            "result.html",
            prediction_label="Invalid Input",
            message="Please provide both email content and sender email.",
            reasons=["Input validation failed: missing content or sender email."],
            is_phishing=True,
            confidence=None,
        )

    features = extract_features(email_text, sender_email)
    rules_phishing, rules_reasons = rule_based_detection(features)

    fallback_used = False
    ml_prediction = 0
    confidence = None
    try:
        ml_prediction, confidence = model_prediction(email_text)
    except Exception:
        # Fallback to keyword/rule detection if model file is unavailable.
        fallback_used = True

    final_is_phishing = rules_phishing or (ml_prediction == 1)
    prediction_label = "Phishing" if final_is_phishing else "Safe"
    message = (
        "⚠️ Warning: This email is likely a phishing attempt!"
        if final_is_phishing
        else "✅ This email appears safe."
    )

    reasons = list(rules_reasons)
    if ml_prediction == 1 and not fallback_used:
        reasons.append("ML model detected phishing-like language patterns.")
    if not reasons and fallback_used:
        reasons.append("Model unavailable. Keyword-based fallback found no strong phishing indicator.")
    if not reasons and not final_is_phishing:
        reasons.append("No suspicious patterns found in content, links, or sender details.")

    return render_template(
        "result.html",
        prediction_label=prediction_label,
        message=message,
        reasons=reasons,
        is_phishing=final_is_phishing,
        confidence=confidence,
        fallback_used=fallback_used,
        link_count=features["link_count"],
        special_char_count=features["special_char_count"],
    )


if __name__ == "__main__":
    app.run(debug=True)
