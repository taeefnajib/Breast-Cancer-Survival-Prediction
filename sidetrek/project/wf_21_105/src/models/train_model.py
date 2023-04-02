# Import all dependencies
from catboost import CatBoostClassifier
import pickle
from src.data.make_dataset import create_train_val
import os

ROOTDIR = os.getcwd()

# Saving the model to disk
def save_model(model):
    filename = f'{ROOTDIR}/models/model.sav'
    pickle.dump(model, open(filename, 'wb'))

def create_model(iterations, loss_function, learning_rate, random_state):
    model = CatBoostClassifier(
    iterations=iterations,
    loss_function=loss_function,
    learning_rate=learning_rate,
    verbose=True,
    random_seed=random_state
    )
    return model

# Fitting model
def fit_model(model, X, y, test_size, random_state):
    X_train, X_valid, y_train, y_valid = create_train_val(X=X, y=y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train,
          eval_set=(X_valid, y_valid),
          verbose_eval=500,
          use_best_model=True
         )
    save_model(model=model)
    return model