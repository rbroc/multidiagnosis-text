#!/bin/bash

python3 baselines.py --run-id beast_1fold_macro_10es --avg-type macro --early-stopping 10 --estimator xgb
python3 baselines.py --run-id beast_1fold_10es --avg-type macro --early-stopping 10 --binary ASD --estimator xgb
python3 baselines.py --run-id beast_1fold_10es --avg-type macro --early-stopping 10 --binary SCHZ --estimator xgb
python3 baselines.py --run-id beast_1fold_10es --avg-type macro --early-stopping 10 --binary DEPR --estimator xgb

python3 baselines.py --run-id beast_1fold_macro_5es --avg-type macro --early-stopping 5 --estimator xgb
python3 baselines.py --run-id beast_1fold_5es --avg-type macro --early-stopping 5 --binary ASD --estimator xgb
python3 baselines.py --run-id beast_1fold_5es --avg-type macro --early-stopping 5 --binary SCHZ --estimator xgb
python3 baselines.py --run-id beast_1fold_5es --avg-type macro --early-stopping 5 --binary DEPR --estimator xgb

python3 baselines.py --run-id beast_1fold_macro_3es --avg-type macro --early-stopping 3 --estimator xgb
python3 baselines.py --run-id beast_1fold_3es --avg-type macro --early-stopping 3 --binary ASD --estimator xgb
python3 baselines.py --run-id beast_1fold_3es --avg-type macro --early-stopping 3 --binary SCHZ --estimator xgb
python3 baselines.py --run-id beast_1fold_3es --avg-type macro --early-stopping 3 --binary DEPR --estimator xgb

python3 baselines.py --run-id beast_1fold_macro_1es --avg-type macro --early-stopping 1 --estimator xgb
python3 baselines.py --run-id beast_1fold_1es --avg-type macro --early-stopping 1 --binary ASD --estimator xgb
python3 baselines.py --run-id beast_1fold_1es --avg-type macro --early-stopping 1 --binary SCHZ --estimator xgb
python3 baselines.py --run-id beast_1fold_1es --avg-type macro --early-stopping 1 --binary DEPR --estimator xgb

python3 baselines.py --run-id rforest --avg-type macro --early-stopping 1 --estimator rforest
python3 baselines.py --run-id rforest --avg-type macro --early-stopping 1 --binary ASD --estimator rforest
python3 baselines.py --run-id rforest --avg-type macro --early-stopping 1 --binary SCHZ --estimator rforest
python3 baselines.py --run-id rforest --avg-type macro --early-stopping 1 --binary DEPR --estimator rforest