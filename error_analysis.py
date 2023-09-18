"""Plot age and sex by whether the ID was predicted correctly or not"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_model_predictions(path: Path):
    """Load model predictions from a jsonl file"""
    return pd.read_json(path, lines=True, orient="records")


def aggregate_predictions(
    df: pd.DataFrame,
    id_col: str,
    predictions_col: str,
    label_col: str,
    idx2label: dict,
) -> pd.DataFrame:
    """Calculates the mean prediction by a grouping col (id_col).
    Args:
        df (pd.DataFrame): Dataframe with 'predictions_col, 'label_col' and `id_col`
        id_col (str): Column to group by
        predictions_col (str): column containing predictions
        label_col (str): column containing labels
        idx2label (dict): mapping from index to label
    Returns:
        pd.DataFrame: Dataframe with aggregated predictions
    """

    def mean_scores(scores: pd.Series):
        gathered = np.stack(scores)
        return gathered.mean(axis=0)

    def get_first_entry(scores: pd.Series):
        return scores.unique()[0]

    df = pd.concat(
        [
            df.groupby(id_col)[predictions_col].apply(mean_scores),
            df.groupby(id_col)[label_col].apply(get_first_entry),
        ],
        axis=1,
    ).reset_index()
    df.columns = ["id", "predictions", "label"]
    df["predicted"] = df["predictions"].apply(lambda x: idx2label[np.argmax(x)])
    return df


def plot_scatter(df, title, x, y, hue, figsize=(10, 4)):
    """Plot scatter"""
    _, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette="colorblind",
        ax=ax,
        legend="full",
    )
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.savefig(f"figs/{x}_{title}.png", dpi=300)
    plt.close()


def plot_gender_histogram(df: pd.DataFrame, title: str):
    """Plot gender histogram"""
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(
        data=df,
        x="Gender",
        hue="correct",
        palette="colorblind",
        ax=ax,
        legend="full",
        multiple="stack",
    )
    ax.set_title(title)
    plt.savefig(f"figs/gender_{title}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    models_dir = Path("logs") / "best_models"
    metadata_path = Path("data") / "splits" / "test_split.csv"
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.rename(columns={"ID": "id"})

    multiclass_mapping = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

    # group by id and get the most common label

    for model_predictions_path in models_dir.glob("*.jsonl"):
        df = load_model_predictions(model_predictions_path)
        df = aggregate_predictions(
            df, "id", "scores", "label", idx2label=multiclass_mapping
        )
        df = df.merge(metadata_df, on="id", validate="1:1")
        df["correct"] = np.where(df["label"] == df["predicted"], "Correct", "Incorrect")

        # plot scatterplot of age colored by correct
        plot_scatter(
            df,
            title=f"age_{model_predictions_path.stem}",
            x="Age",
            y="correct",
            hue="correct",
        )

        # plot histogram of gender colored by correct
        plot_gender_histogram(df, model_predictions_path.stem)

        # plot scatterplot of HamD colored by correct
        plot_scatter(
            df,
            title=f"hamd_{model_predictions_path.stem}",
            x="DepressionSeverity",
            y="correct",
            hue="label",
        )
        # plot scatterplot of ADOS colored by correct
        plot_scatter(
            df,
            title=f"ados_{model_predictions_path.stem}",
            x="AutismSeverity",
            y="correct",
            hue="label",
        )
