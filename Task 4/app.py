
# In[196]:


#Load the necessary Libraries
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import pickle


# In[209]:

# Load the dataset
data = pd.read_csv('train.csv')

# Separate features and target variable
X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp_model = MLPClassifier(random_state=42)
mlp_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

ensemble_model = VotingClassifier(estimators=[
    ('mlp', mlp_model),
    ('dt', dt_model)
], voting='hard')
ensemble_model.fit(X_train, y_train)

# In[210]

with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp_model, f)

with open('dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

with open('mlp_model.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('dt_model.pkl', 'rb') as f:
    model2 = pickle.load(f)

with open('ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)


# In[212]:


app = FastAPI()


# In[211]:


class PredictionResponse(BaseModel):
    predicted_cover_type: int
    predictions: list[int]

# In[212]

# Define the root (index) endpoint
@app.get("/")
def read_root():
    return {"hello": "Welcome to the Forest Cover Type Prediction API"}


# In[213]:


# Define the version endpoint
@app.get("/version")
def read_version():
    return {"version": "1.0"}


# In[215]:


@app.post("/predict/mlp")
def predict_mlp(data: dict):
    try:
        features = np.array(list(data.values())).reshape(1, -1)
        predictions = mlp_model.predict(features)
        predicted_cover_type = int(predictions[0])
        return PredictionResponse(predicted_cover_type=predicted_cover_type, predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# In[216]:


@app.post("/predict/decision-trees")
def predict_decision_trees(data: dict):
    try:
        features = np.array(list(data.values())).reshape(1, -1)
        predictions = dt_model.predict(features)
        predicted_cover_type = int(predictions[0])
        return PredictionResponse(predicted_cover_type=predicted_cover_type, predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# In[218]:


@app.post("/predict/ensemble")
def predict_ensemble(input_data):
    try:
        features = np.array(list(input_data.dict().values())).reshape(1, -1)
        predictions = ensemble_model.predict(features)
        predicted_cover_type = int(predictions[0])
        return PredictionResponse(predicted_cover_type=predicted_cover_type, predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# In[220]:


import uvicorn


# In[221]:


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000)









