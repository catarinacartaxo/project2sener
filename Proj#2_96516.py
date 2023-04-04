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
import plotly.graph_objects as go

#kbest
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)
#Emsemble methods
from sklearn.ensemble import RandomForestRegressor


#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#Load data
df = pd.read_csv('testData_2019_Central.csv') #load meteo data
#columns #'Date', 'Central (kWh)', 'temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'solarRad_W/m2', 'rain_mm/h', 'rain_day

df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type


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



#--- create dataframe of 2019 for model
#--- add holdidays/weekends and power-1
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
y2 = np.delete(y2,-1) #delete last row to match number of lines

#-- CHOOSE VARIABLES
#df_data2019_clean=df_data2019.drop(columns=['temp_C','HR','windSpeed_m/s','windGust_m/s','pres_mbar','rain_mm/h','rain_day'])
df_clean_2019=df_data2019.drop(columns=['HR','windSpeed_m/s','windGust_m/s','pres_mbar','solarRad_W/m2','rain_mm/h','rain_day'])
df_clean_2019=df_clean_2019.iloc[:, [0,1,5,2,3,4]] # Change the position of the columns so that Y=column 0 and X all the remaining columns


# // ---- RANDOM FOREST WITH 4 FEATURES

#Load RF model
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

fig_RF = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:3])



#//----- GET 2017-18 DATA----\\
    

df_1718_raw = pd.read_csv('all_clean_1718.csv') #load meteo data

#get pow-1 for 17/18
df_1718_clean=df_1718_raw
df_1718_clean['power-1']=df_1718_clean['Power_kW'].shift(1)
df_1718_clean=df_1718_clean.dropna()

#2,3,4,5,6,7,8,9,10,11,12
X1=df_1718_clean.iloc[:,2:]
Y1=df_1718_clean.iloc[:,1]


X1_train=X1.values
Y1_train=Y1.values




# // ---- RANDOM FOREST WITH 3 FEATURES

X3_train=df_1718_clean.iloc[:,[2,10,11]] #fatures 2017/18
Y3_train=Y1_train #real pow 2017/18
X3=df_data2019.iloc[:,[2,10,11]]

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model2 = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model2.fit(X3_train, Y3_train)
y_pred_RF4 = RF_model2.predict(X3)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y_pred_RF4) 
MSE_RF=metrics.mean_squared_error(y2,y_pred_RF4)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y_pred_RF4))
cvRMSE_RF=RMSE_RF/np.mean(y2)

#plot
d={'Date':df_real['Date'].values, 'RandomForest': y_pred_RF4}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')

fig_RF4 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:3])


#metrics 
d = {'Methods': 'Random-Forest - Withouth Power-1', 'MAE': MAE_RF, 'MSE': MSE_RF, 'RMSE': RMSE_RF, 'cvMSE': cvRMSE_RF}
df_metrics = df_metrics.append(d, ignore_index = True)


#make linear regression with all variavles
from sklearn import  linear_model

# // ---- LINEAR REGRESSION WITH 4 FEATURES

#x_axis from 2019 - is X2
X4_train=df_1718_clean.iloc[:,[12,2,10,11]] #fatures 2017/18
Y4_train=Y1_train #real pow 2017/18


# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets with 1718
regr.fit(X4_train,Y4_train)
# Make predictions using the testing set 2019
y_pred_LR4 = regr.predict(X2)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y_pred_LR4) 
MSE_LR=metrics.mean_squared_error(y2,y_pred_LR4)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y_pred_LR4))
cvRMSE_LR=RMSE_LR/np.mean(y2)

#plot
d={'Date':df_real['Date'].values, 'LinearRegression': y_pred_LR4}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')

fig_LR4 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:3])

#metrics 
d = {'Methods': 'Linear Regression', 'MAE': MAE_LR, 'MSE': MSE_LR, 'RMSE': RMSE_LR, 'cvMSE': cvRMSE_LR}
df_metrics = df_metrics.append(d, ignore_index = True)


# // ---- LINEAR REGRESSION WITH 3 FEATURES


# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets with 1718
regr.fit(X3_train,Y3_train)
# Make predictions using the testing set 2019
y_pred_LR3 = regr.predict(X3)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y_pred_LR3) 
MSE_LR=metrics.mean_squared_error(y2,y_pred_LR3)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y_pred_LR3))
cvRMSE_LR=RMSE_LR/np.mean(y2)

#plot
d={'Date':df_real['Date'].values, 'LinearRegression': y_pred_LR3}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')

fig_LR3 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:3])

#metrics 
d = {'Methods': 'Linear Regression - Withouth Power-1', 'MAE': MAE_LR, 'MSE': MSE_LR, 'RMSE': RMSE_LR, 'cvMSE': cvRMSE_LR}
df_metrics = df_metrics.append(d, ignore_index = True)


# // ---- LINEAR REGRESSION WITH all FEATURES

X_LR=df_data2019.drop(columns=['Date','Central (kWh)']) #removes date (x) and power
X_LR2=X_LR.values

regr = linear_model.LinearRegression()
# Train the model using the training sets with 1718
regr.fit(X1_train,Y1_train)
# Make predictions using the testing set 2019
y_pred_LR = regr.predict(X_LR2)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y2,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)

#plot
d={'Date':df_real['Date'].values, 'LinearRegression': y_pred_LR}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')

fig_LR = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:3])

#add metrics do table
d = {'Methods': 'Linear Regression - All Features', 'MAE': MAE_LR, 'MSE': MSE_LR, 'RMSE': RMSE_LR, 'cvMSE': cvRMSE_LR}
df_metrics = df_metrics.append(d, ignore_index = True)


#choises of models and respective features        
available_models=['Random Forest','Linear Regression']
available_features={'Random Forest': ['Power-1, Temperature, Hour of Day, Holiday/Weekend','Temperature, Hour of Day, Holiday/Weekend'], 'Linear Regression': ['Power-1, Temperature, Hour of Day, Holiday/Weekend','Temperature, Hour of Day, Holiday/Weekend','All']}



#//-------- EDA -----\\
    
#get data      
df_1718_features=df_1718_clean.columns[2:]

#radio item to choose between hist or graph
radioitem=dcc.RadioItems(id='radio',options=['Graph','Histogram'],value='Graph')

# Create a dropdown menu to select columns to include in the GRAPH
column_options = [{'label': col, 'value': col} for col in df_1718_features]
dropdown_graph = dcc.Dropdown(id='dropdown_graph',options=column_options, multi=True, value='temp_C')

#options of features for histo
# Create a dropdown menu to select columns to include in the HIST
column_options = [{'label': col, 'value': col} for col in df_1718_features]
dropdown_hist = dcc.Dropdown(id='dropdown_hist',options=column_options, value='temp_C')

#default graph is temp
fig_graph = px.line(df_1718_clean, x='Date_start', y=df_1718_clean.columns[1:3])

#default histogram is temp
fig_hist=go.Figure()
fig_hist.add_trace(go.Histogram(x=df['temp_C'],name='temperature'))


#// ---  FEATURE SELECTION -----\\
# Define input and outputs

Y=Y1_train
X=X1_train
#X= df_1718_features.values

#// -- KBest --- \\
#features=SelectKBest(k=2,score_func=f_regression) # Test different k number of features, uses f-test ANOVA
features=SelectKBest(k=3,score_func=f_regression)

fit=features.fit(X,Y) #calculates the scores using the score_function f_regression of the features
features_results=fit.transform(X)

features=SelectKBest(k=2,score_func=mutual_info_regression) # Test different k number of features, uses f-test ANOVA
#features=SelectKBest(k=3,score_func=mutual_info_regression) # Test different k number of features, uses f-test ANOVA

fit=features.fit(X,Y) #calculates the f_regression of the features
features_results=fit.transform(X)




#default histogram is temp
fig_hist=go.Figure()
fig_hist.add_trace(go.Histogram(x=df['temp_C'],name='temperature'))

#// -- Emsemble Method --- \\\

model = RandomForestRegressor()
model.fit(X, Y)
print(model.feature_importances_) # Power-1 hour Temp


d={'Features': X1.columns.values,'KBest': fit.scores_,'EmsM':model.feature_importances_}
df_features_scores=pd.DataFrame(data=d)


figKbest = px.bar(df_features_scores,x="Features", y='KBest')
figEmsm = px.bar(df_features_scores,x="Features", y='EmsM')



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

#// --- APP --- \\
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
    html.H2('IST Central Building Energy Forecast'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Metrics', value='tab-3'),
        dcc.Tab(label='EDA', value='tab-4'),
        dcc.Tab(label='Feature Selection', value='tab-5'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST 2019 Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig_raw_data,
            ),
            
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('IST 2019 Electricity Forecast (kWh)'),
            'Choose forecast model',
            dcc.Dropdown(
                id='menu',
                options=[{'label': i, 'value': i} for i in available_models],
                value='Random Forest',
            ),
            html.Br(),
            'Choose features',
            html.Div(dcc.Dropdown(id='dropdown_ft'), id='dropdown_ft_container'),
            dcc.Graph(
                id='forecast',
                ),
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Electricity Forecast Errors'),
            generate_table(df_metrics)
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Visualization of features of forecast model (2017/2018)'),
            radioitem,
            html.Div([],id='radio-contents')
            ])
    elif tab == 'tab-5':
        return  html.Div([
            html.H3('Importance of features'),
            dcc.Dropdown(
                id='dropdown_sel_met',
                options=['KBest','Emsemble Method'],
                value='KBest'),
            dcc.Graph(id='graph_sel_met',)
            ])
 
    """
    dcc.Graph(id='graph_kbest',figure=figKbest),
    html.H4('KBest'),
    dcc.Graph(id='graph_kbest',figure=figEmsm),
    html.H4('Emsemble Methods')
    """
    

#CALLBACK TO CHANGE BETWEEN forecast MODELS
@app.callback(
    dash.dependencies.Output('forecast', 'figure'),
    [dash.dependencies.Input('menu', 'value'),dash.dependencies.Input('dropdown_ft', 'value')])
def update_graph(model,ftrs):
    if model == 'Random Forest':
        if ftrs == 'Power-1, Temperature, Hour of Day, Holiday/Weekend':
            return fig_RF
        if ftrs == 'Temperature, Hour of Day, Holiday/Weekend':
            return fig_RF4
        
    elif model == 'Linear Regression':
        if ftrs == 'All':
            return fig_LR
        elif ftrs == 'Power-1, Temperature, Hour of Day, Holiday/Weekend':
            print('LR 4')
            return fig_LR4
        elif ftrs == 'Temperature, Hour of Day, Holiday/Weekend':
            print('LR 3')
            return fig_LR3


@app.callback(
    Output('dropdown_ft_container', 'children'),
    Input('menu', 'value')
)
def set_neighborhood(model):
    model_features = available_features[model]
    return dcc.Dropdown(model_features, model_features[0], id='dropdown_ft',
    )

#//--- EDA ---\\
#//---------------------------------------------------------
#//---------------------------------------------------------

#callback for RADIO item
@app.callback(
    Output('radio-contents', 'children'),
    Input('radio', 'value')
)
def render_content_radio(value):
    if value == 'Graph':
        return html.Div([
            dropdown_graph,
            dcc.Graph(id='graphG', figure=fig_graph)
            ])
    elif value == 'Histogram':
        return html.Div([
            html.H6('Hist'),
            dropdown_hist,
            dcc.Graph(id='graphH', figure=fig_hist)
            ])    
    
# Set up the callback to upDate the forecast graph when the dropdown is changed
@app.callback(
    Output('graphG', 'figure'),
    Input('dropdown_graph', 'value')
)
def update_graph2(selected_columns):
    y_axis=['Power_kW']
    for i in selected_columns:
        y_axis.append(i)
    fig_features = px.line(df_1718_clean, x='Date_start', y=y_axis)
   # fig = px.line(df, x='Date', y=df.columns[1],label=y_axis)
   ## for col in selected_columns:
    #    fig.add_line(x=df['Date'], y=df[col], name=col)
    return fig_features

# Set up the callback to upDate the Histogram when the dropdown is changed
@app.callback(
    Output('graphH', 'figure'),
    Input('dropdown_hist', 'value')
)
def update_graphH(value):
    fig= go.Figure()
    fig.add_trace(go.Histogram(x=df_1718_clean[value],name=value))
    print(value)
    return fig

# Set up the callback for feature selection method
@app.callback(
    dash.dependencies.Output('graph_sel_met', 'figure'),
    [dash.dependencies.Input('dropdown_sel_met', 'value')])
def update_graph_FS(value):
    if value == 'KBest':
        return figKbest
    elif value == 'Emsemble Method':
        return figEmsm

if __name__ == '__main__':
    app.run_server()



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
