import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import os
import joblib
# return models[model_name](**model_params)
# Things we need to define as variables to run the functions (can be easily customized to what is required) 
model_name=""
model_params={}
model_save_path=""

def Creating_training_data(csv_file, num_feature_columns, Index_Of_Target_Value):
    data_file = pd.read_csv(csv_file)
    Y_train = data_file[Index_Of_Target_Value]
    X_train = data_file.drop(columns=num_feature_columns, axis=1)

    return X_train, Y_train
    
def get_training_model(model_name, model_params):
    models = {
        "RandomForestRegressor": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor
        # We can add more models if needed 
    }
    if model_name not in models:
        print("The "+str(model_name)+" model is not found.")
    return models[model_name](**model_params)

def training_rfmodel(X_train,Y_train,model_save_path):
    rf_model = get_training_model(model_name, model_params)
    rf_model.fit(X_train, Y_train)
    joblib.dump(rf_model, model_save_path)
    print(" The Model is saved to" +str(model_save_path))
    return rf_model
    

def retrain(X_New,Y_New,model_save_path):
    if os.path.exists(model_save_path):
       rf_model = joblib.load(model_save_path)
       rf_model.fit(X_New, Y_New)
       joblib.dump(rf_model, model_save_path)
       print(" New Model saved to" +str(model_save_path))
       return rf_model
    else:
        print("Error, path doesn't exist")











