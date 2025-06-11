import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from gda import GDAClassifier
from sklearn.metrics import confusion_matrix
import warnings

def main():
    warnings.filterwarnings("ignore")

    #   banknote authentication data
    X, y = fetch_openml("banknote-authentication", version=1, as_frame=False, return_X_y=True)
    y = y.astype(int)

    print(f"Dataset shape: {X.shape}, Classes: {np.unique(y)}")

    # Normalize  
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # tts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # train GDA
    model = GDAClassifier()
    model.fit(X_train, y_train)

    #  predict
    y_pred = model.predict(X_test)
    acc = model.accuracy(y_test, y_pred)

    print(f"\nGDA Accuracy on Banknote Authentication: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nSample predictions:")
    for i in range(10):
        print(f"True: {y_test[i]}, Predicted: {y_pred[i]}")


if __name__ == "__main__":
    main()
