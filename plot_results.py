"""Create the two figures produced by the notebook for the selected case."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_case(city_metrics, training_history, output_dir):
    ax = city_metrics.set_index("Test_city")[["Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(7, 5))
    ax.set(ylim=(0, 1), xlabel="Independent test city", ylabel="Score", title="Leave-one-city-out U-Net performance")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(output_dir / "typical_case_precision_recall_f1.png", dpi=300); plt.close()
    fig, ax = plt.subplots(figsize=(8, 5))
    for stage, frame in training_history.groupby("Stage"):
        ax.plot(frame["Epoch"], frame["Train_loss"], linewidth=2, label=stage)
    ax.set(xlabel="Epoch", ylabel="Training loss", title=f"Test city {city_metrics.iloc[0]['Test_city']}")
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(output_dir / "typical_case_training_loss.png", dpi=300); plt.close(fig)

