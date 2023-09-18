import pandas as pd
from pathlib import Path

from psycopmlutils.model_performance import performance_metrics_from_folder

if __name__ == "__main__":

    log_path = Path("logs")
    baselines = log_path / "baselines" / "json_outs"
    transformers = log_path / "transformers" / "json_outs"

    dfs = []

    for target_class in ["DEPR", "ASD", "SCHZ", "multiclass"]:
        if target_class != "multiclass":
            id2label = {0: "TD", 1: target_class}
        else:
            id2label = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

        for p in [baselines, transformers]:
            perf = performance_metrics_from_folder(
                p,
                pattern=f"*{target_class}*.jsonl",
                id_col="id",
                id2label=id2label,
                metadata_cols="all",
            )
            dfs.append(perf)
    dfs = pd.concat(dfs)
    dfs.to_json("text_performance.jsonl", orient="records", lines=True)
