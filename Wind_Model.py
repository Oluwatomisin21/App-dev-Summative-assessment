import pandas as pd  
import json
import pickle
import urllib.request, urllib.parse, urllib.error 
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# get_ipython().run_line_magic('matplotlib', 'inline')


import pandas as pd
import numpy as np

data1 = pd.read_csv ('wind_farm.csv')
data2 = pd.read_csv ('wind_generation_data.csv',sep=",")

data2
#print(data2.astype(int))
#data_2=data2.astype(int)

data2.fillna(value=0, inplace=True) 
data2.info()
data2.head()
# splitting variaables 
#X = data_train2(['wind speed','direction'], axis = 1).values # X are the input (or independent) variables
y = data2['Power Output'].values # Y is output (or dependent) variable
X = data2.drop(['Power Output'] , axis=1).values


# X = x.reshape(-1,1)
# y = y.reshape(-1,1)

# Dropping any rows with Nan values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.linear_model import LinearRegression 

# Splitting the data into training and testing data 
regr = LinearRegression() 
  
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test)) 

y_pred = regr.predict(X_test)

print(y_pred)

json_string = urllib.request.urlopen('https://api.openweathermap.org/data/2.5/onecall?lat=53.5615&lon=8.5907&exclude=hourly,minutely&appid=289a66a9c614883cbcc27b672fc390b3').read()
weatherdata = json.loads(json_string)

 

#Let's throw this data into a dataframe so we can take a better look at it
weatherdata2 = pd.DataFrame(weatherdata['daily'])



weatherdata2
W22=weatherdata2[['dt','wind_speed','wind_deg']]

# W22

W33=W22.rename(columns={"wind_speed": "wind speed", "wind_deg": "direction"})

# W33

W22['dt']



W22['dt'] = pd.to_datetime(W22['dt'],unit='s')

W22.head()



WZ=W22[['wind_speed','wind_deg']]
WZ
x1=WZ.values

y_pred = regr.predict(x1)

print(y_pred)

# 1.append ypredict power to w22,date,wind,direction.
# 2. extract day from date Epoch conversion
# 3. develop a logic 0.7*7 = 4.9 MW

# e = np.random.normal(size=10) 
df=pd.DataFrame(y_pred, columns=['Predicted Power Output']) 
print (df)

dfz =W22.join(df, lsuffix='_caller', rsuffix='_other')
dfz

# dfz2 =dfz.join(df, lsuffix='_caller', rsuffix='_other')
# A continuous index value will be maintained 
# across the rows in the new appended data frame. 
# df1 =df.append(W22,ignore_index = True) 

# dfz
dfz
# ### WIND DATA
# dfz = pd.merge(round_data, wind_data_renamed, on=['Day'])
# dfz['Predicted Power Output'] = dfz['Predicted Power Output'] * dfz['Capacity'] /100

pickle.dump(regr,open('WIND_FARM_MODEL.pkl','wb'))
model= pickle.load(open('WIND_FARM_MODEL.pkl','rb'))

