from pathlib import Path

import pandas as pd
from psycopmlutils.model_comparison import ModelComparison

if __name__ == "__main__":

    transformers_folder = Path("logs") / "transformers" / "json_outs"
    baselines_folder = Path("logs") / "baselines" / "json_outs"

    metadata_cols = [
        "model_name",
        "split",
        "type",
        "binary",
        "target_class",
        "is_baseline",
    ]

    dfs = []
    # binary
    for diagnosis in ["DEPR", "ASD", "SCHZ"]:
        score_mapping = {0: "TD", 1: diagnosis}
        model_comparer = ModelComparison(
            id_col="id", score_mapping=score_mapping, metadata_cols=metadata_cols
        )
        transformer_df = model_comparer.transform_data_from_folder(
            transformers_folder, pattern=f"{diagnosis}*.jsonl"
        )
        baseline_df = model_comparer.transform_data_from_folder(
            baselines_folder, pattern=f"*{diagnosis}*.jsonl"
        )
        dfs.append(transformer_df)
        dfs.append(baseline_df)

    # multiclass
    multiclass_mapping = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}
    model_comparer = ModelComparison(
        id_col="id", score_mapping=multiclass_mapping, metadata_cols=metadata_cols
    )
    dfs.append(
        model_comparer.transform_data_from_folder(
            transformers_folder, pattern="multiclass*.jsonl"
        )
    )
    dfs.append(
        model_comparer.transform_data_from_folder(
            baselines_folder, pattern="*multiclass*.jsonl"
        )
    )

    dfs = pd.concat(dfs)
    dfs.to_json("text_performance.jsonl", orient="records", lines=True)
