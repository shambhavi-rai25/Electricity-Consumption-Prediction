# Electricity-Consumption-Prediction
Electricity Consumption Prediction Using Regression
A regression project that predicts electricity consumption using weather and time features, with a Streamlit web app for live predictions.

### Dataset: Power Consumption of Tetouan City (Kaggle)
Link: https://www.kaggle.com/datasets/ruchikakumbhar/power-consumption-prediction-dataset

The dataset records power use of three distribution zones every 10 minutes for one year, along with weather readings.

52,416 rows, 9 columns, readings every 10 minutes

### Models Used
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest

### Evaluation Metrics 
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R2 Score

### Best Model
Random Forrest based on R2 Score for target variables(Zone 1, Zone 2, Zone 3, Total Consumption)

### Tech Stack
- Google Colab
- Python
- Pandas/Numpy
- Scikit-learn
- SciPy
- Matplotlib/Seaborn
- Streamlit

### Streamlit app
Enter temperature, humidity, wind speed, general diffuse flows, diffuse flows, day of week, hour, month and day of month, then click Predict Electricity Consumption. 
The app shows the predicted total and the Zone 1, Zone 2 and Zone 3 consumption in kW, with a bar chart and a table of the inputs.
