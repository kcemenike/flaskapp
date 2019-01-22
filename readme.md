# API Documentation - Plant Model #

This API has been designed as a microservice to predict, retrain and get prediction parameters from the plant machine learning model.


## Predict ##

Parse parameters as json. E.g. to predict the Fuel consumption for a power Generation value of 4000, use:

```python
powerGen = {"powerGen":4000}
response = requests.post("{}/predict".format(BASE_URL), json=powerGen) #The BASE_URL is usually servername:5000 ("http://127.0.0.1:5000")
response.json()
```

## Retrain with new data ##

Retrain model with new data. Parse data as json (if in CSV, you need to convert to JSON)

```python
data = json.dumps([{"Power Generated":4664.80,"Operating Hours":24,"Fuel Consumed":918900},
                   {"Power Generated":43170,"Operating Hours":24,"Fuel Consumed":883904}])
response = requests.post("{}/retrain".format("http://127.0.0.1:5000"), json = data)
response.json()
```

## Get Model Parameters ##

Get model parameters

response = requests.get("{}/getParameters".format("http://127.0.0.1:5000"))
response.json()