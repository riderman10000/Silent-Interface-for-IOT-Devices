import requests
import time

# URL = "http://localhost:5000/getEmg"
URL = "https://silent-app-test.herokuapp.com/getEmg"
received = False
while(not received):
    res = requests.get(url= URL)
    receivedData = res.json()
#     print(receivedData)
    print(".",end="")
    if receivedData['result'] == 'None':
        received = False
    else:
        received = True
# print(receivedData)
#this data should be filtered and pass to ml model 
receivedData = np.array(receivedData['result'])
print(receivedData.shape)
data_feature = feature_pipeline(receivedData)
# data_feature.shape



reshape_feature = np.zeros((data_feature.shape[0], data_feature.shape[2],
                         data_feature.shape[3], data_feature.shape[1]))
for i in range(data_feature.shape[0]):
    for j in range(8):
        reshape_feature[i,:,:,j] = data_feature[i,j,:,:]

data_feature = reshape_feature
print(f"After Reshape: {data_feature.shape}")
del reshape_feature


prediction = model.predict_classes(data_feature)
stringPrediction = list(label_encoder.inverse_transform(list(prediction)))
print(stringPrediction)



payload = {'sentence' : stringPrediction[0]}
r = requests.post(URL, json=payload)