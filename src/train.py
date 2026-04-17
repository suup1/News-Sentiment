import pandas as pd
import nltk
import mlflow
import mlflow.sklearn
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

from mlflow.tracking import MlflowClient

# =========================
# PATH CONFIG (ROBUST)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "financial_news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# =========================
# MLFLOW CONFIG
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Default")

# Download NLTK safely
nltk.download('stopwords', quiet=True)

MODEL_NAME = "sentiment-model"


# =========================
# LOAD DATA
# =========================
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(
        path,
        encoding="latin1",
        header=None,
        names=["label", "text"]
    )
    return df


# =========================
# PREPROCESS
# =========================
def preprocess(df):
    return df["text"], df["label"]


# =========================
# PIPELINE
# =========================
def build_pipeline(alpha):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB(alpha=alpha))
    ])


# =========================
# TRAIN + LOG
# =========================
def train_and_log(alpha, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=f"NB_alpha_{alpha}") as run:

        mlflow.set_tag("model_type", "NaiveBayes")

        model = build_pipeline(alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("accuracy", accuracy)

        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        report_path = f"report_alpha_{alpha}.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODEL_DIR, f"model_alpha_{alpha}.pkl"))

        print(f"Alpha={alpha} | Accuracy={accuracy:.4f}")

        return accuracy, run.info.run_id, model


# =========================
# PROMOTION FUNCTION
# =========================
def promote_best_model(best_run_id, best_accuracy, threshold=0.65):

    client = MlflowClient()

    if best_accuracy < threshold:
        print("Model NOT promoted (below threshold)")
        return

    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=result.version
    )

    print(f"Model version {result.version} promoted to @production")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    best_accuracy = 0
    best_run_id = None
    best_alpha = None
    best_model = None

    for alpha in [0.5, 1.0, 2.0]:
        acc, run_id, model = train_and_log(alpha, X_train, X_test, y_train, y_test)
        results[alpha] = acc

        if acc > best_accuracy:
            best_accuracy = acc
            best_run_id = run_id
            best_alpha = alpha
            best_model = model

    print(f"\nBEST MODEL: alpha={best_alpha} with accuracy={best_accuracy:.4f}")

    # =========================
    # FINAL DEPLOYMENT MODEL
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)

    final_model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    joblib.dump(best_model, final_model_path)

    print(f"Final deployment model saved at {final_model_path}")

    # PROMOTE BEST MODEL
    promote_best_model(best_run_id, best_accuracy)