# weather_prediction

🟦 1. Data Loading and Inspection
pd.read_csv("weather.csv", index_col="DATE"): Reads a CSV file, using the "DATE" column as the index.

weather.head(), weather.dtypes, weather.index: Basic data inspection.

🟦 2. Null Value Handling
weather.apply(pd.isnull).sum()/weather.shape[0]: Calculates percentage of missing values in each column.

Drops columns with more than 5% missing data:
valid_columns = weather.columns[null_pct < .05]

weather.ffill(): Forward-fills missing data (useful for time series).

🟦 3. Data Cleaning
weather.columns.str.lower(): Standardizes column names to lowercase.

Converts index to datetime:
weather.index = pd.to_datetime(weather.index)

🟦 4. Feature Engineering
Target Variable: Creates a new column target representing next day's max temperature (tmax).

Rolling averages & percent change for features like tmax, tmin, prcp over 3 and 14-day horizons.

python
Copy
Edit
weather[col].rolling(horizon).mean()
Seasonal Averages: Adds monthly and daily averages using .groupby() and .expanding().mean().

🟦 5. Predictive Modeling
Uses Ridge Regression (Ridge(alpha=.1)) to model the target.

Predictors exclude target, name, station.

🟦 6. Backtesting Function
backtest() splits the data into rolling training and testing windows.

Trains and tests the model iteratively in steps (e.g., 90 days).

Calculates absolute prediction differences and returns results over time.

🟦 7. Evaluation
Uses mean_absolute_error() to evaluate model performance.

Plots prediction error distribution:
predictions["diff"].round().value_counts().sort_index().plot()

🟦 8. Plotting and Visualization
Plots snow depth (snwd) and analyzes prediction differences over time.

