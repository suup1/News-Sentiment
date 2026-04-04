import pandas as pd
import nltk
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

from mlflow.tracking import MlflowClient

# =========================
# CONFIG
# =========================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Default")

nltk.download('stopwords')

DATA_PATH = "data/financial_news.csv"
MODEL_NAME = "sentiment-model"


# =========================
# LOAD DATA
# =========================
def load_data(path=DATA_PATH):
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
def train_and_log(alpha):
    with mlflow.start_run(run_name=f"NB_alpha_{alpha}") as run:

        df = load_data()
        X, y = preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = build_pipeline(alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        # LOG PARAMS + METRICS
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("accuracy", accuracy)

        # CLASSIFICATION REPORT
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        report_path = "report.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)

        # LOG MODEL (no auto register here)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print(f"Alpha={alpha} | Accuracy={accuracy:.4f}")

        return accuracy, run.info.run_id


# =========================
# PROMOTION FUNCTION
# =========================
def promote_best_model(best_run_id, best_accuracy, threshold=0.65):

    client = MlflowClient()

    if best_accuracy < threshold:
        print("Model NOT promoted (below threshold)")
        return

    # Register model manually from best run
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    # Set alias (NEW MLFLOW WAY)
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

    results = {}
    best_accuracy = 0
    best_run_id = None
    best_alpha = None

    for alpha in [0.5, 1.0, 2.0]:
        acc, run_id = train_and_log(alpha)
        results[alpha] = acc

        if acc > best_accuracy:
            best_accuracy = acc
            best_run_id = run_id
            best_alpha = alpha

    print(f"\nBEST MODEL: alpha={best_alpha} with accuracy={best_accuracy:.4f}")

    # PROMOTE BEST MODEL
    promote_best_model(best_run_id, best_accuracy)