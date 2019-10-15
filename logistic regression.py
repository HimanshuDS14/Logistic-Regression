import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Social_Network_Ads.csv")
print(data.head(10))

x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

scaler = StandardScaler()
scaler.fit(x)
scaled_feature = scaler.transform(x)

print(scaled_feature)

train_x,test_x,train_y,test_y = train_test_split(scaled_feature , y , test_size=0.3 , random_state=0)

log = LogisticRegression()
log.fit(train_x , train_y)
pred = log.predict(test_x)
print(pred)
print(test_y)
print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))


#for prediction
z = np.array([26,43000])
z = z.reshape(1,-1)

scaler1 = StandardScaler()
scaled = scaler.transform(z)

print(log.predict(scaled))



