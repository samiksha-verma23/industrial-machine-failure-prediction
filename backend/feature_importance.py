from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "machine_failure_model.pkl"
OUTPUT_PATH = BASE_DIR / "models" / "feature_importance.csv"


model = joblib.load(MODEL_PATH)


preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["model"]


feature_names = preprocessor.get_feature_names_out()

importances = random_forest.feature_importances_


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})


importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


importance_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nFeature Importance:")
print(importance_df.to_string(index=False))

print(f"\nSaved to: {OUTPUT_PATH}")