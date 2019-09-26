
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import *
from sklearn.metrics import *


#import missingno as msno
# https://medium.com/ibm-data-science-experience/missing-data-conundrum-exploration-and-imputation-techniques-9f40abe0fd87


# In[269]:


# Functions & calculations

# Calculate Tukey's fences for outliers
def calculate_outlier_fences(set):
    Q1 = np.percentile(set, 25)
    Q3 = np.percentile(set, 75)
    IQR = Q3 - Q1
    C = 1.5
    L_Fence = (Q1 - (C * IQR)).astype(float)
    U_Fence = (Q3 + (C * IQR)).astype(float)
    
    return [L_Fence, U_Fence]

# Return outliers based on the fences
def spot_outliers(set):    
    for i in set:
        if i < calculate_outlier_fences(set)[0]:
            print("Datapoint ", i, " is an outlier. The lower fence is ", variable_fences(set)[0])
        elif i > calculate_outlier_fences(set)[1]:
            print("Datapoint ", i, " is an outlier. The upper fence is ", variable_fences(set)[1])
            
def calculate_r_squared(y, y_hat):
#     y = [y]
#     y_hat = [y_hat]
    zipped = zip(y, y_hat)
    mean = np.mean(y)
    up = 0
    down = 0
    for i,j in zipped:
        
        up = up + ((i-j)**2)
        down = down + ((i - mean)**2)
   #print(1 - (up/down))
   #print("up = ", up, ", down = ", down)
    return (1 - (up/down))


# In[78]:


df = pd.read_csv("GW_dataset.csv")
df.tail()


# In[79]:


#define variables
Temperature = df["Temperature"]
CO2_A = df["CO2_Atmosphere"]
Rainfall = df["Rainfall"]
Year = df["Year"]
CO2_E = df["CO2_Emissions"]

#check correlations between variables
df.corr() #correlation between temperature & CO2_Atmoshpere, Temperature & Rainfall


# In[80]:


#Check mean + std
print(df.describe())


# In[90]:


#Treat missing data
# https://medium.com/ibm-data-science-experience/missing-data-conundrum-exploration-and-imputation-techniques-9f40abe0fd87


# In[84]:


#Make linear regression model for Temperature
fit = np.polyfit(Year, Temperature, 1)
fit_fn = np.poly1d(fit) 

#Save fences for temperature
L_Fence_Temp = calculate_outlier_fences(df["Temperature"])[0]
U_Fence_Temp = calculate_outlier_fences(df["Temperature"])[1]


# In[184]:


#Get the slope of the linear regression model
simple_lm_temp_slope = fit_fn[1]
simple_lm_temp_interc = fit_fn[0]

#Calculate temperature in 2030
def calc_simple_lm_temp(year):
    temperature = simple_lm_temp_slope * year + simple_lm_temp_interc
    print(temperature)
    
#Calculate for 2030
calc_simple_lm_temp(2030)


# In[185]:


#Plot regression model Temperature + upper and lower fence
plt.plot(Year, Temperature, 'yo', Year, fit_fn(Year), '--k')
plt.xlim(1960, 2030)
plt.ylim(12,24)
plt.axhline(y = L_Fence_Temp, color='black', linestyle='-')
plt.axhline(y = U_Fence_Temp, color='black', linestyle='-')


# In[148]:


train, test = train_test_split(df, test_size = 0.2, random_state = 3, stratify = None)

train = train.sort_values("Year")
test = test.sort_values("Year")

#model_df = df.loc[:, df.columns == "Year", "Rainfall":"CO2_Emissions"]
features = ["Year", "Rainfall"]
target = "Temperature"
train.head()


# In[187]:


# set linear regression model
lr = LinearRegression()

# fit the regression model on the features and the predicted feature
lr.fit(train[features], train[target])

# put results in variables
data = train[target]
predict = lr.predict(train[features])

# compute error
training_error = mean_absolute_error(data, predict)

# set list with real target data
test_data = test[target]

# set list with predicted target data
predict_test = lr.predict(test[features])

# test for error between real and predicted target data
test_data_error = mean_absolute_error(test_data, predict_test)


# In[199]:


results = pd.DataFrame(test["Year"])
results["Real_data"] = test_data
results["Predicted_data"] = predict_test.round(2)
results["Error"] = (test_data - predict_test).round(2)

results


# In[181]:


plt.plot(results["Year"], results["Real_data"], 'ko', results["Year"], results["Predicted_data"], 'yo')


# In[270]:


calculate_r_squared(results["Real_data"], results["Predicted_data"])


# In[267]:


calculate_r_squared(results["Real_data"], results["Predicted_data"])

