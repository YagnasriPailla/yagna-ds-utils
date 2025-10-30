from pathlib import Path
import joblib
import pandas as pd
from typing import Union

def predict_from_joblib(model_path: Union[str, Path],
                        features_csv: Union[str, Path]) -> float:
    """
    Load a joblib model and a 1-row CSV of features, return prediction as float.
    """
    model = joblib.load(model_path)
    df = pd.read_csv(features_csv)
    pred = float(model.predict(df)[0])
    return pred
