import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_results():
    results_path = "results/results.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError("Run evaluate.py first to generate results.json")

    with open(results_path, "r") as f:
        data = json.load(f)

    samples = data["samples"]
    accuracy = data["accuracy"]

    actual = [s["actual"] for s in samples]
    predicted = [s["predicted"] for s in samples]
    correct = [s["correct"] for s in samples]

    os.makedirs("results", exist_ok=True)

    # 1. Correct vs Incorrect
    plt.figure(figsize=(5, 4))
    counts = Counter(correct)
    plt.bar(["Correct", "Incorrect"], [counts[True], counts[False]])
    plt.title(f"Accuracy: {accuracy:.2%}")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig("results/accuracy_bar.png")
    plt.close()

    # 2. Answer Distribution
    truth_dist = Counter(actual)
    pred_dist = Counter(predicted)
    labels = ["A", "B", "C", "D", "E"]

    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    plt.bar(x, [truth_dist[l] for l in labels], width=0.4, label="Actual", align="center")
    plt.bar([p + 0.4 for p in x], [pred_dist[l] for l in labels], width=0.4, label="Predicted", align="center")
    plt.xticks([p + 0.2 for p in x], labels)
    plt.xlabel("Answer Choice")
    plt.ylabel("Frequency")
    plt.title("Answer Distribution (Actual vs Predicted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/answer_distribution.png")
    plt.close()

    # 3. Confusion Matrix
    cm = confusion_matrix(actual, predicted, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print(" Saved all plots to /results/")

if __name__ == "__main__":
    plot_results()
