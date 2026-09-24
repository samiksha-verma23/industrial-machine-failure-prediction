from pathlib import Path

import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "machine_failure_model.pkl"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

VALID_RANGES = {
    "Air temperature [K]": (295, 305),
    "Process temperature [K]": (305, 314),
    "Rotational speed [rpm]": (1168, 2886),
    "Torque [Nm]": (3.8, 76.6),
    "Tool wear [min]": (0, 253)
}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Industrial Machine Failure Prediction API is running"
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No input data received."
            }), 400

        missing_fields = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        if data["Type"] not in ["L", "M", "H"]:
            return jsonify({
                "status": "error",
                "message": "Machine Type must be L, M, or H."
            }), 400

        input_data = {
            "Type": data["Type"],
            "Air temperature [K]": float(data["Air temperature [K]"]),
            "Process temperature [K]": float(data["Process temperature [K]"]),
            "Rotational speed [rpm]": float(data["Rotational speed [rpm]"]),
            "Torque [Nm]": float(data["Torque [Nm]"]),
            "Tool wear [min]": float(data["Tool wear [min]"])
        }

        for feature, (minimum, maximum) in VALID_RANGES.items():

            value = input_data[feature]

            if value < minimum or value > maximum:
                return jsonify({
                    "status": "error",
                    "message": f"{feature} must be between {minimum} and {maximum}."
                }), 400

        dataframe = pd.DataFrame([input_data])

        prediction = int(model.predict(dataframe)[0])

        probabilities = model.predict_proba(dataframe)[0]

        failure_probability = float(probabilities[1])
        normal_probability = float(probabilities[0])

        if prediction == 1:
            result = "Machine Failure Predicted"
        else:
            result = "No Machine Failure Predicted"

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "result": result,
            "failure_probability": round(
                failure_probability * 100, 2
            ),
            "normal_probability": round(
                normal_probability * 100, 2
            )
        })

    except ValueError:
        return jsonify({
            "status": "error",
            "message": "Invalid input values. Please enter valid numbers."
        }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():

    try:
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("numerical__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
            .str.replace("Type_", "Type ", regex=False)
        )

        return jsonify({
            "status": "success",
            "features": importance_df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )