


import pandas as pd  
import numpy as np 
import pickle
import json
import urllib.request, urllib.parse, urllib.error 
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# get_ipython().run_line_magic('matplotlib', 'inline')

data1 = pd.read_csv ('solar_farm.csv')
data2 = pd.read_csv ('solar_generation_data.csv',sep=",")

data2
#print(data2.astype(int))
#data_2=data2.astype(int)

data2_fill = data2.dropna(0)

dat2 = data2[['Temp Hi','Temp Low','Solar','Cloud Cover Percentage','Rainfall in mm','Power Generated in MW']]
data2_fill



data2_fill.info()
data2_fill

data2_fill.info()


data2_fill['Temp Hi'] = data2_fill['Temp Hi'].replace('\u00b0','', regex=True).astype(float)
data2_fill['Temp Low'] = data2_fill['Temp Low'].replace('\u00b0','', regex=True).astype(float)

# data22.info()

#X = data_train2(['wind speed','direction'], axis = 1).values # X are the input (or independent) variables
y = data2_fill['Power Generated in MW'].values # Y is output (or dependent) variable
X = data2_fill.drop(['Power Generated in MW'] ,  axis=1).values

# Test and split values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


from sklearn.linear_model import LinearRegression 

# Splitting the data into training and testing data 
regr = LinearRegression() 
  
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test))

y = regr.predict(X)

# print(y)



json_string = urllib.request.urlopen('https://api.openweathermap.org/data/2.5/onecall?lat=53.5615&lon=8.5907&exclude=hourly,minutely&appid=289a66a9c614883cbcc27b672fc390b3').read()
weatherdata = json.loads(json_string)

#Let's throw this data into a dataframe so we can take a better look at it
weatherdata2 = pd.DataFrame(weatherdata['daily'])

weatherdata2
# type(weatherdata2)


www=weatherdata2['temp']
www

FXP= pd.DataFrame(weatherdata2['temp'].values.tolist(), index=weatherdata2.index)

FXP


FXP1=FXP[['min','max']]







dfz =FXP1.join(W22, lsuffix='_caller', rsuffix='_other')


dfz






import datetime

dy=weatherdata2[['dt']]
dy['dt'] = pd.to_datetime(dy['dt'],unit='s')

dy

W33=dfz.rename(columns={"min": "Temp Hi", "max": "Temp Low","uvi":"solar", "clouds":"Cloud Cover Percentage", "rain":"Rainfall in mm"})


# W33

W44 = W33[['Temp Hi','Temp Low','solar','Cloud Cover Percentage','Rainfall in mm']]
W44



x1=W44.values

y= regr.predict(x1)
print(y)
# e = np.random.normal(size=10) 
# df=pd.DataFrame(y_pred, columns=['Power Output']) 
# print (df)



# e = np.random.normal(size=10) 
df=pd.DataFrame(y, columns=['Predicted Power Output']) 
print (df)


dfz =W44.join(df, lsuffix='_caller', rsuffix='_other')

dfz

dfz

dfz1 =dy.join(dfz, lsuffix='_caller', rsuffix='_other')
dfz1


# Logic



# ## SOLAR DATA
# new_solar = pd.merge(round_data, solar_data_renamed, on=['Day'])
# new_solar['Predicted_SolarFarm_Output(MW)'] = new_solar['Predicted_SolarFarm_Output(MW)'] * new_solar['Capacity'] /100
# final_solar = pd.merge(round_data, new_solar, how='outer', on='Day', suffixes=('_',''))
# final_solar_data = final_solar.drop(['Capacity'], axis=1)
# new_cols_ = final_solar_data.columns[final_solar_data.columns.str.endswith('_')]



# x = np.arange(10)

# condlist = [x<3, x>5]

# choicelist = [x, x**2]

# np.select(condlist, choicelist)
# array([ 0,  1,  2, ..., 49, 64, 81])


pickle.dump(regr,open('SOLAR_FARM_MODEL.pkl','wb'))

model= pickle.load(open('SOLAR_FARM_MODEL.pkl','rb'))