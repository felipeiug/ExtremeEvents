# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:59:45 2024

@author: bbren
"""
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).
    y_true: torch.Tensor of actual/observed values
    y_pred: torch.Tensor of predicted values
    """
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()  # Convert tensor to scalar value

def NSE(y_true, y_pred):
    """
    Compute the R-squared (coefficient of determination).
    y_true: torch.Tensor of actual/observed values
    y_pred: torch.Tensor of predicted values
    """
    # Calculate residuals and their sum of squares
    residuals = y_true - y_pred
    ss_res = torch.sum(residuals ** 2)

    # Calculate total sum of squares
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)

    # Compute R-squared
    r2 = 1 - (ss_res / ss_tot)

    return r2.item()  # Convert tensor to a scalar value


class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.ReLU
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# ==========================LOAD DATA =======================================
#Loading Demand Data

demandFile = 'data/DemandData2.csv'  
dfDemand= pd.read_csv(demandFile)

#Loading Weather Data

WeatherFile = 'data/WeatherData2.csv'  
dfWeather = pd.read_csv(WeatherFile)

# Ensure the "Date-time CET-CEST" columns are in datetime format for both DataFrames
dfDemand['Date-time CET-CEST (DD/MM/YYYY HH:mm)'] = pd.to_datetime(dfDemand['Date-time CET-CEST (DD/MM/YYYY HH:mm)'])
dfWeather['Date-time CET-CEST (DD/MM/YYYY HH:mm)'] = pd.to_datetime(dfWeather['Date-time CET-CEST (DD/MM/YYYY HH:mm)'])

# Get the start and end dates from dfDemand
start_date = dfDemand['Date-time CET-CEST (DD/MM/YYYY HH:mm)'].min()
end_date = dfDemand['Date-time CET-CEST (DD/MM/YYYY HH:mm)'].max()

# Filter dfWeather to match the date range from dfDemand
filtered_weather_df_interval = dfWeather[
    (dfWeather['Date-time CET-CEST (DD/MM/YYYY HH:mm)'] >= start_date) & 
    (dfWeather['Date-time CET-CEST (DD/MM/YYYY HH:mm)'] <= end_date)]
# Display the filtered weather DataFrame


# ==========================HANDLING DATA =======================================
#Get the DMA of interest and also the time horizon of interest

ForecastingDMA = 'DMA A'
ForecastingValues = dfDemand[ForecastingDMA]
ForecastingHorizon = 168 #in hours
OtherDMAs = dfDemand.drop(columns=[ForecastingDMA])
start, end = 0, ForecastingHorizon  # Example range

# Exclude rows from index start to end at the forecasting 
ForecastingValues = ForecastingValues.drop(ForecastingValues.index[start:end])

# Remove the same number of rows from the second DataFrame, starting from the end for input
num_rows_to_exclude = end - start 
WeatherVariables = filtered_weather_df_interval.iloc[:-num_rows_to_exclude]
OtherDMAs = OtherDMAs.iloc[:-num_rows_to_exclude]

# ==========================INPUT DATA =======================================


# Create a new DataFrame with separate columns for day, month, and hour
dfCalendar = pd.DataFrame({
    'Hour': WeatherVariables['Date-time CET-CEST (DD/MM/YYYY HH:mm)'].dt.hour,
    'DayOfWeek' : WeatherVariables['Date-time CET-CEST (DD/MM/YYYY HH:mm)'].dt.dayofweek
})

# Exclude the first column of dfWeather
dfWeather_excluded = WeatherVariables.drop(columns=['Date-time CET-CEST (DD/MM/YYYY HH:mm)'])
OtherDMAs_excluded = OtherDMAs.drop(columns=['Date-time CET-CEST (DD/MM/YYYY HH:mm)'])
DemandExcluded =  dfDemand.drop(columns=['Date-time CET-CEST (DD/MM/YYYY HH:mm)'])
# Concatenate dfWeather_split and the remaining columns of dfWeather
dfWeather_combined = pd.concat([dfCalendar,dfWeather_excluded,OtherDMAs_excluded], axis=1)

# Assume ForecastingValues is a pandas DataFrame or Series
# Ensure that ForecastingValues and dfWeather_combined are aligned by index
X = dfWeather_combined.to_numpy()  # Convert input dataset to NumPy array
y = ForecastingValues.to_numpy()   # Convert target dataset to NumPy array

split_index = int(0.8 * len(X))

# Split sequentially
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure it's of shape [batch_size, 1]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # Ensure it's of shape [batch_size, 1]


num_batches=1
batch_size=12823
x_aux=[]
y_aux=[]
#

batchx=[]
batchy=[]

batchxPressure=[]
batchxFlow = []
batchTrainData=[]

for i in range(num_batches):
  # Get starting and ending indices for the current batch
  start_idx = i * batch_size
  end_idx = start_idx + batch_size
  #x_tensor_aux = torch.tensor(x_aux,dtype=torch.float32)
  #y_tensor_aux = torch.tensor(y_aux,dtype=torch.float32)
  # Extract the current batch data
  batch_datax = X_train_tensor[start_idx:end_idx]
  batch_datay = y_train_tensor[start_idx:end_idx]
  batchx.append(batch_datax)
  batchy.append(batch_datay)
  
  
  
# ==========================Build and Train the forecasting model=======================================

#3
NNeurons =  [32,128,256]
NLayers =[2,3,4]
N_INPUT = dfWeather_combined.shape[1]  # Number of features (columns in dfWeather_combined)
N_OUTPUT = 1  # Assuming you are predicting a single value at a time

RTrain =np.zeros((len(NNeurons), len(NLayers)))
MAETrain=np.zeros((len(NNeurons), len(NLayers)))

RValid=np.zeros((len(NNeurons), len(NLayers)))
MAEValid=np.zeros((len(NNeurons), len(NLayers)))

num_epochs = 1000  # You can adjust this based on performance

for neurons in range(len(NNeurons)):
    for layers in range(len(NLayers)):
        model = FCN(N_INPUT,N_OUTPUT,NNeurons[neurons],NLayers[layers])

        # Dynamic Learning rate
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        # scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.5)
       
        # Define early stopping parameters
        early_stopping_patience = 10
        best_loss = 10**12
        epochs_without_improvement = 0
           
        for i in range(num_epochs):
            for nbatch in range((num_batches)):
                optimizer.zero_grad()

                loss=0
                outputs = model(batchx[nbatch])

                # Neural Network Loss Functions
                loss = (torch.mean((outputs-batchy[nbatch])**2))

                loss.backward()
                optimizer.step()
                
            # Dynamic learning rate
            if i == 75:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-3  # Change learning rate
            elif i == 100:
                      for param_group in optimizer.param_groups:
                          param_group['lr'] = 5e-4  # Change learning rate
            elif i == 125:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4  # Change learning rate
            elif i == 500:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4  # Change learning rate
            elif i == 2000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-4  # Change learning rate

            predictions = model(X_test_tensor)
            lossvvalid = (torch.mean((predictions-y_test_tensor)**2))
                                                                    
            print("Epoch:",(i)," lr:",optimizer.param_groups[0]['lr']," Global:",(lossvvalid.detach().numpy())**(0.5))

            
        # Evaluate the model on the test data
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions = model(X_test_tensor)
            
        MAETrain[neurons,layers] = mean_absolute_error(y_train_tensor, outputs)
        RTrain[neurons,layers]=NSE(y_train_tensor,outputs)
        MAEValid[neurons,layers] = mean_absolute_error(y_test_tensor, predictions)
        RValid[neurons,layers]=NSE(y_test_tensor,predictions)
        with open(f'results/demand/Model_A_168{NLayers[layers]}_{NNeurons[neurons]}.pkl', 'wb') as f:
            pickle.dump(model, f)  
        



