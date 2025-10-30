# yagna-ds-utils

Tiny helper to run a prediction from a saved joblib model.

```python
from yagna_ds_utils import predict_from_joblib
y = predict_from_joblib("models/solana_rf.joblib","models/last_row_features_solana.csv")
print(y)
