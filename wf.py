# Importing all dependencies
from src.data.make_dataset import make_dataset
from src.models.train_model import create_model, fit_model
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from catboost import CatBoostClassifier
import os
import joblib
import pathlib

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "data/raw/patient_data.csv"
    test_size: float = 0.2
    random_state: int = 26
    iterations: int = 5000
    loss_function: str = 'Logloss'
    learning_rate: float = 0.0001

hp = Hyperparameters()

# Running workflow
def run_wf(hp: Hyperparameters) ->CatBoostClassifier:
    X, y = make_dataset((pathlib.Path(__file__).parent / hp.filepath).resolve())
    model = create_model(hp.iterations, hp.loss_function, hp.learning_rate, hp.random_state)
    return fit_model(model=model, X=X, y=y, test_size=hp.test_size, random_state=hp.random_state)

if __name__=="__main__":
    run_wf(hp=hp)