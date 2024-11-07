# Setup
- using linux or wsl is recommended (from my experience btw)
- create a virtual enviornment using python (<3.11, >3.9)
  
  ```python3.11 -m venv venv```
  
  or (using conda)
  
  ```conda create -n python==3.11 venv```
- activate it
  
  ```source venv\bin\activate ``` (linux)
- change the directory
  
  ```cd Used-Car-Price-Prediction-endToEnd```
- install ```requiremnets.txt```
  
  ```pip install -r requirements.txt```
- Now for ```zenml``` use wsl or linux and run ```zenml init```
- then ```zenml up```
- for the default one run (in vs code)
  
  ```zenml connect --url=http://127.0.0.1:8237```
- then run the pipeline
  
  ```python pipelines/training_pipeline.py```

- for mlflow interactive ui

   ```mlflow ui```
