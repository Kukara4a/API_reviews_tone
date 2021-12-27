import requests
import pandas as pd
import json

data = pd.DataFrame(pd.read_pickle('test.csv'), columns=['Text'])

request = requests.post("http://localhost:5000/predict", data.to_json())
print(request.status_code)

response = json.loads(request.content)
response_df = pd.DataFrame(response, columns=['Text', 'Prediction'])
response_df.to_csv('response.csv', index=False)
print(pd.read_csv('response.csv').head())