import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import holidays
from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data
df = pd.read_csv('testData_2019_Central.csv') #load meteo data

#columns #'Date', 'Central (kWh)', 'temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'solarRad_W/m2', 'rain_mm/h', 'rain_day

df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type

print(df.columns)
#///------RAW DATA TAB-----\\\
#get dataframe with meteo 2019 data
#df_meteo=df.drop(columns='Central (kWh)') #removes date (x) and power

df2=df.iloc[:,2:]
fig_raw_data = px.line(df, x="Date", y=df2.columns)


#//----TAB 2----\\
    
#get real power    
#set dataframe to make model
#make prediciton
#plot prediction and real power

#get real power
df_real=df.iloc[:,0:2]
y2=df['Central (kWh)'].values
#2)set dataframe
df_data2019=df






pt_lis_holidays = holidays.country_holidays('PT', subdiv='11', years=[2019]) 
df_data2019['Holiday'] = (df_data2019['Date'].isin(pt_lis_holidays)).astype(float) #if it's holiday: =1.0

print(df_data2019.dtypes)
#create columns for Day of Week and for Hour of the day
df_data2019['Week_Day'] = df_data2019['Date'].dt.dayofweek
df_data2019['Hour']=df_data2019['Date'].dt.hour

#replace day of week for Weekends
df_data2019['Weekend'] = df_data2019['Week_Day'].apply(lambda x: 1.0 if(x >= 5) else 0.0)
df_data2019=df_data2019.drop(columns=['Week_Day'])

# replace with column that says if it's holiday or weekend
df_data2019['Hol_Wknd'] = 0.0
df_data2019.loc[(df_data2019['Weekend'] == 1.0) | (df_data2019['Holiday'] == 1.0), 'Hol_Wknd'] = 1.0
df_data2019=df_data2019.drop(columns=['Weekend','Holiday'])

df_data2019['power-1']=df_data2019['Central (kWh)'].shift(1)

#from the 2018 power dataframe we can replace the power-1 for the first hour of 01-01-2019
df_data2019.loc['2019-01-01 00:00:00', 'power-1'] = 109.9247786

df_data2019=df_data2019.dropna()
y2 = np.delete(y2,-1)

#df_data2019_clean=df_data2019.drop(columns=['temp_C','HR','windSpeed_m/s','windGust_m/s','pres_mbar','rain_mm/h','rain_day'])
df_data2019_clean=df_data2019.drop(columns=['HR','windSpeed_m/s','windGust_m/s','pres_mbar','solarRad_W/m2','rain_mm/h','rain_day'])
df_data2019_clean=df_data2019_clean.iloc[:, [0,1,5,2,3,4]] # Change the position of the columns so that Y=column 0 and X all the remaining columns
df_clean_2019=df_data2019_clean
df_clean_2019

#Load LR model
with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)
    
# recurrent
Z=df_clean_2019.values
Y=Z[:,1]
X2=Z[:,[2,3,4,5]]

y2_pred_RF = RF_model.predict(X2)


fig_raw_data = px.line(df, x="Date", y=df2.columns)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)

d = {'Methods': ['Random Forest'], 'MAE': [MAE_RF], 'MSE': [MSE_RF], 'RMSE': [RMSE_RF], 'cvMSE': [cvRMSE_RF]}
df_metrics = pd.DataFrame(data=d)

df_real = df_real[:-1]

d={'Date':df_real['Date'].values, 'RandomForest': y2_pred_RF}
df_forecast=pd.DataFrame(data=d)

df_results=pd.merge(df_real,df_forecast, on='Date')

fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:3])

"""



#APPPPPPPPPPPPPPPPPPPPPP WITH ONLY THE FIRST GRAPH
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('IST Energy Monitor - Dashboard 1'),
    dcc.Graph(
        id='yearly-data',
        figure=fig_raw_data,
    ),
    dcc.Graph(
        id='yearly-data',
        figure=fig2,
    ),
    
])

if __name__ == '__main__':
    app.run_server(debug=False)
"""
# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H2('IST Energy Forecast tool (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Table', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig_raw_data,
            ),
            
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
                ),
            generate_table(df_metrics)
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('IST Electricity Forecast (kWh)'),
            generate_table(df_metrics)
        ])


if __name__ == '__main__':
    app.run_server()



"""
#Create matrix from data frame
Y=df_pow1718.values
#Identify output Y

#Identify input X
#X=Z[:,[1,3,5,6]]
X=df_met1718.values
#X=Z[:,[1,5,6]] -->only 3 isn't very accurate for more than the next hour after 2018
#X=Z[:,[1,5]] --> isn't good

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)
#print(X_train)
#print(y_train)


from sklearn import  linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)

df_results=
fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])


"""



"""

#Create matrix from data frame
Z=df_data.values
#Identify output Y
Y=Z[:,0]
#Identify input Y
#X=Z[:,[1,3,5,6]]
X=Z[:,[1,2,5,6]]
#X=Z[:,[1,5,6]] -->only 3 isn't very accurate for more than the next hour after 2018
#X=Z[:,[1,5]] --> isn't good

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)
#print(X_train)
#print(y_train)
"""

"""
#Load data
df = pd.read_csv('forecast_data.csv')
df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type
#df = df.set_index('Date') # make 'datetime' into index
#df.rename(columns = {'Power-1':'power', 'Day week':'day'}, inplace = True)
df2=df.iloc[:,1:5]
X2=df2.values
fig = px.line(df, x="Date", y=df.columns[1:4])


df_real = pd.read_csv('real_results.csv')
y2=df_real['Power (kW) [Y]'].values

#Load and run models

with open('LR_model.pkl','rb') as file:
    LR_model2=pickle.load(file)

y2_pred_LR = LR_model2.predict(X2)



#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)



#Load RF model
with open('RF_model.pkl','rb') as file:
    RF_model2=pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)

d = {'Methods': ['Linear Regression','Random Forest'], 'MAE': [MAE_LR, MAE_RF], 'MSE': [MSE_LR, MSE_RF], 'RMSE': [RMSE_LR, RMSE_RF],'cvMSE': [cvRMSE_LR, cvRMSE_RF]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'LinearRegression': y2_pred_LR,'RandomForest': y2_pred_RF}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')


fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


"""