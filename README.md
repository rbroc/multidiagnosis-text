### Multi-class diagnosis prediction using text

Code for training feature-based baselines and Transformer models for classification of mental disorders from text, both in a multiclass classification setting and in a binary classification setting.

- `train_baselines.py` contains code to train a variety of feature-based XGboost baselines, and baselines based on static vectors;
- `train_transformer.py` contains code to train transformer-based classifiers;
- `train_classifier.py` contains code to run baselines as classifications with a shallow neural network.
- `preproc.py` and `dataset.py` are dataset processing utils (data not shared due to confidentiality)

The repository also includes additional utils and notebook for exploration of the results.

Models and their predictions are logged under `logs`. Aggregate model performance is reported in `logs/aggregates`, which also contains results from speech-based models.
The twin repository for audio-based models is here: https://github.com/HLasse/multidiagnosis-speech

Related publication (available [here](https://arxiv.org/abs/2301.06916)):
Hansen, L., Rocca, R., Simonsen, A., Parola, A., Bliksted, V., Ladegaard, N., ... & Fusaroli, R. (2023). Automated speech-and text-based classification of neuropsychiatric conditions in a multidiagnostic setting. arXiv preprint arXiv:2301.06916.