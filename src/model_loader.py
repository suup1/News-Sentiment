import mlflow
import mlflow.pyfunc


def load_production_model():
    """
    Load model from MLflow Model Registry (Production stage)
    """

    # Set tracking URI (SQLite backend)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Load model from registry
    model = mlflow.pyfunc.load_model(
        model_uri="models:/sentiment-model/Production"
    )

    return model


if __name__ == "__main__":
    # Load model
    model = load_production_model()

    # Sample input
    sample = ["Stock market is crashing badly"]

    # Prediction
    prediction = model.predict(sample)

    print("Prediction:", prediction)