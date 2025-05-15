import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
from datetime import datetime, timedelta
from scipy.optimize import minimize
import holidays

st.set_page_config(page_title="Uber Trip Explorer 🚕", layout="wide")
st.title("🚕 Uber Trip Explorer: Uncover Ride Trends & Predict Demand! 🌟")
st.markdown("""
Explore Uber trip patterns and forecast future rides with ease! 📅 Pick a date to see predictions, view trends, and get actionable insights. No tech skills needed! 😊
""")

st.sidebar.header("🚀 Get Started!")
st.sidebar.markdown("""
1. 📂 Enter your Uber trip CSV file path.
2. 🤖 Choose a prediction model.
3. 📅 Select a future date for forecasts.
4. 📈 Adjust test data size.
5. 🎉 View insights and predictions below!
""")
data_path = st.sidebar.text_input("📂 CSV File Path", value="uber-data.csv", help="Path to your Uber trip CSV, e.g., 'C:/data/uber.csv'")
prediction_tool = st.sidebar.selectbox("🤖 Prediction Model", ["Random Forest", "XGBoost", "Gradient Boosting", "Ensemble"], 
                                      help="Choose a model to predict trips. 'Ensemble' combines all for top accuracy! 💪")
future_date = st.sidebar.date_input("📅 Pick a Future Date", min_value=datetime.today(), max_value=datetime.today() + timedelta(days=365), 
                                   help="Select a date to predict trips for the week! 🕒")
test_percentage = st.sidebar.slider("📈 Test Data Size (%)", 10, 40, 20, 5, help="Percentage of data for testing predictions. 🔍")

@st.cache_data
def load_and_prepare_data(file_path):
    st.info("⏳ Loading your Uber trip data...")
    try:
        df = pd.read_csv(r"C:\Users\pc\OneDrive\Documents\cip 2025\Data Analyst\uber\Uber-Jan-Feb-FOIL.csv")
    except FileNotFoundError:
        st.error("⚠️ File not found! Check the path and try again.")
        return None
    
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df.sort_values('date', inplace=True)
    
    df = df.groupby('date').agg({'trips': 'sum', 'active_vehicles': 'sum'}).reset_index()
    df.set_index('date', inplace=True)
    
    day_labels = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['dayofweek'] = df.index.dayofweek
    df['day_name'] = df['dayofweek'].map(day_labels)
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    
    df['trips_lag1'] = df['trips'].shift(1)
    df['trips_lag7'] = df['trips'].shift(7)
    df['vehicles_lag1'] = df['active_vehicles'].shift(1)
    df['rolling_avg_trips_7d'] = df['trips'].rolling(window=7).mean()
    df['rolling_avg_vehicles_7d'] = df['active_vehicles'].rolling(window=7).mean()
    df.fillna(method='ffill', inplace=True)
    
    st.success("✅ Data loaded and ready to roll!")
    return df

def create_past_data(data, target='trips'):
    past_days = 14  # Use 2 weeks of daily data
    X, y = [], []
    for i in range(len(data) - past_days):
        X.append(data.iloc[i:i + past_days][['trips', 'active_vehicles', 'trips_lag1', 'trips_lag7', 'vehicles_lag1', 
                                             'rolling_avg_trips_7d', 'rolling_avg_vehicles_7d', 'dayofweek', 'is_weekend', 'is_holiday']].values.flatten())
        y.append(data.iloc[i + past_days][target])
    return np.array(X), np.array(y)

@st.cache_resource
def train_prediction_tool(X_train, y_train, tool_choice):
    st.info(f"⚙️ Training {tool_choice} model...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.7, 0.9, 1.0]
    }
    gbr_param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.7, 0.9, 1.0]
    }
    
    tools = {
        'Random Forest': (RandomForestRegressor(random_state=42), rf_param_grid),
        'XGBoost': (xgb.XGBRegressor(objective='reg:squarederror', random_state=42), xgb_param_grid),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=42), gbr_param_grid)
    }
    
    tool, param_grid = tools.get(tool_choice, tools['Random Forest'])
    grid_search = GridSearchCV(estimator=tool, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    st.success(f"🎉 {tool_choice} model trained!")
    return grid_search

def optimize_ensemble_weights(xgb_pred, rf_pred, gbr_pred, y_true):
    def objective(weights):
        combined = weights[0] * xgb_pred + weights[1] * rf_pred + weights[2] * gbr_pred
        return mean_absolute_percentage_error(y_true, combined)
    
    initial_weights = [0.33, 0.33, 0.34]
    constraints = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
    bounds = [(0, 1)] * 3
    result = minimize(objective, initial_weights, constraints=constraints, bounds=bounds)
    return result.x

def combine_predictions(xgb_pred, rf_pred, gbr_pred, weights):
    return weights[0] * xgb_pred + weights[1] * rf_pred + weights[2] * gbr_pred

def prepare_future_data(historical_data, future_date):
    past_days = 14  # Use 2 weeks of daily data
    future_date = pd.to_datetime(future_date)
    future_days = pd.date_range(start=future_date, end=future_date + timedelta(days=6), freq='D')
    
    last_data = historical_data.iloc[-past_days:].copy()
    
    future_df = pd.DataFrame(index=future_days)
    future_df['date'] = future_df.index
    day_labels = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['day_name'] = future_df['dayofweek'].map(day_labels)
    future_df['month'] = future_df.index.month
    future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
    us_holidays = holidays.US()
    future_df['is_holiday'] = future_df.index.map(lambda x: 1 if x in us_holidays else 0)
    
    future_df['trips'] = last_data['trips'].mean()
    future_df['active_vehicles'] = last_data['active_vehicles'].mean()
    future_df['trips_lag1'] = last_data['trips_lag1'].mean()
    future_df['trips_lag7'] = last_data['trips_lag7'].mean()
    future_df['vehicles_lag1'] = last_data['vehicles_lag1'].mean()
    future_df['rolling_avg_trips_7d'] = last_data['rolling_avg_trips_7d'].mean()
    future_df['rolling_avg_vehicles_7d'] = last_data['rolling_avg_vehicles_7d'].mean()
    
    X_future = []
    for i in range(len(future_days)):
        start_idx = max(0, i - past_days + len(last_data))
        past_data = pd.concat([last_data, future_df.iloc[:i]]).tail(past_days)
        X_future.append(past_data[['trips', 'active_vehicles', 'trips_lag1', 'trips_lag7', 'vehicles_lag1', 
                                  'rolling_avg_trips_7d', 'rolling_avg_vehicles_7d', 'dayofweek', 'is_weekend', 'is_holiday']].values.flatten())
    
    return np.array(X_future), future_df

if data_path:
    df = load_and_prepare_data(data_path)
    
    if df is not None:
        st.subheader("📊 Data Snapshot")
        st.markdown("Here's a peek at your Uber trip data! 👀")
        st.dataframe(df[['trips', 'active_vehicles', 'day_name', 'is_weekend', 'is_holiday']].head(), use_container_width=True)
        
        st.subheader("🔢 Key Metrics")
        total_trips = df['trips'].sum()
        unique_days = df.index.nunique()
        avg_trips_per_day = df['trips'].mean()
        avg_vehicles = df['active_vehicles'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚗 Total Trips", f"{total_trips:,}", help="Total Uber rides in the dataset.")
        col2.metric("📅 Days Covered", unique_days, help="Number of days with data.")
        col3.metric("🚕 Avg Trips/Day", f"{avg_trips_per_day:.0f}", help="Average daily trips.")
        col4.metric("🛠️ Avg Vehicles", f"{avg_vehicles:.0f}", help="Average active vehicles daily.")
        
        st.subheader("📈 Trip Patterns")
        st.markdown("Discover when Uber rides peak! 🌆")
        
        day_labels = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        fig_day = px.histogram(df, x='day_name', y='trips', title='🚖 Busiest Days of the Week',
                              labels={'day_name': 'Day of Week', 'trips': 'Number of Trips'},
                              color_discrete_sequence=['#EF553B'], category_orders={'day_name': list(day_labels.values())})
        fig_day.update_layout(bargap=0.1)
        st.plotly_chart(fig_day, use_container_width=True)
        
        peak_day = df.groupby('day_name')['trips'].mean().idxmax()
        peak_trips = df.groupby('day_name')['trips'].mean().max()
        st.markdown(f"🌟 **{peak_day}** is the busiest with ~{peak_trips:.0f} trips on average! Schedule extra drivers! 🚕")
        
        fig_month = px.histogram(df, x='month', y='trips', title='📅 Trips by Month',
                                labels={'month': 'Month', 'trips': 'Number of Trips'},
                                color_discrete_sequence=['#00CC96'])
        fig_month.update_layout(bargap=0.1)
        st.plotly_chart(fig_month, use_container_width=True)
        
        peak_month = df.groupby('month')['trips'].mean().idxmax()
        peak_month_trips = df.groupby('month')['trips'].mean().max()
        st.markdown(f"📈 **Month {peak_month}** sees ~{peak_month_trips:.0f} trips on average. Plan for high demand! 🏙️")
        
        st.subheader("🔄 Trip Trends Over Time")
        st.markdown("See how trips evolve with trends and weekly patterns! 📉")
        result = seasonal_decompose(df['trips'], model='additive', period=7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=result.observed, name='Actual Trips', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=df.index, y=result.trend, name='Trend', line=dict(color='#00CC96')))
        fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, name='Weekly Pattern', line=dict(color='#EF553B')))
        fig.add_trace(go.Scatter(x=df.index, y=result.resid, name='Irregular Changes', line=dict(color='#AB63FA')))
        fig.update_layout(title='Trip Patterns Over Time', xaxis_title='Date', yaxis_title='Number of Trips')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        - **Actual Trips**: Real daily trip counts. 📊
        - **Trend**: Shows if trips are rising or falling. 📈
        - **Weekly Pattern**: Highlights busy days like weekends. 🔄
        - **Irregular Changes**: Spots unusual spikes (e.g., holidays). 🎉
        """)
        
        X_trip, y_trip = create_past_data(df, target='trips')
        cutoff_idx = int(len(X_trip) * (1 - test_percentage/100))
        X_train_trip, X_test_trip = X_trip[:cutoff_idx], X_trip[cutoff_idx:]
        y_train_trip, y_test_trip = y_trip[:cutoff_idx], y_trip[cutoff_idx:]
        test_dates = df.index[14 + cutoff_idx:]
        
        if prediction_tool != "Ensemble":
            trip_grid = train_prediction_tool(X_train_trip, y_train_trip, prediction_tool)
            best_tool_trip = trip_grid.best_estimator_
            trip_pred = best_tool_trip.predict(X_test_trip)
        else:
            xgb_trip = train_prediction_tool(X_train_trip, y_train_trip, "XGBoost").best_estimator_
            rf_trip = train_prediction_tool(X_train_trip, y_train_trip, "Random Forest").best_estimator_
            gbr_trip = train_prediction_tool(X_train_trip, y_train_trip, "Gradient Boosting").best_estimator_
            
            xgb_pred_trip = xgb_trip.predict(X_test_trip)
            rf_pred_trip = rf_trip.predict(X_test_trip)
            gbr_pred_trip = gbr_trip.predict(X_test_trip)
            
            trip_weights = optimize_ensemble_weights(xgb_pred_trip, rf_pred_trip, gbr_pred_trip, y_test_trip)
            trip_pred = combine_predictions(xgb_pred_trip, rf_pred_trip, gbr_pred_trip, trip_weights)
        
        st.subheader("🔮 Prediction Performance")
        st.markdown(f"How well does the **{prediction_tool}** model predict trips? 🤔")
        
        accuracy_trip = r2_score(y_test_trip, trip_pred)
        error_percentage_trip = mean_absolute_percentage_error(y_test_trip, trip_pred)
        error_size_trip = root_mean_squared_error(y_test_trip, trip_pred)
        
        st.markdown(f"""
        - **Accuracy (R²)**: {accuracy_trip:.3f} (closer to 1 = better) ✅
        - **Error Percentage**: {error_percentage_trip:.2%} (lower = better) 📉
        - **Average Error**: {error_size_trip:.0f} trips (lower = better) 🎯
        **Takeaway**: High accuracy means reliable predictions for planning! 🚀
        """)
        
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test_dates, y=y_test_trip, name='Actual Trips', line=dict(color='#636EFA')))
        fig_test.add_trace(go.Scatter(x=test_dates, y=trip_pred, name='Predicted Trips', line=dict(color='#EF553B', dash='dash')))
        fig_test.update_layout(title=f'{prediction_tool} Predictions vs Actual', xaxis_title='Date', yaxis_title='Number of Trips')
        st.plotly_chart(fig_test, use_container_width=True)
        
        st.markdown("**Blue line**: Actual trips. **Red dashed line**: Predictions. Close match = trustworthy forecasts! 😎")
        
        st.subheader(f"🔮 Predictions for Week of {future_date.strftime('%B %d, %Y')}")
        st.markdown(f"See expected trips for the week starting {future_date.strftime('%B %d, %Y')}! 📅")
        
        X_future, future_df = prepare_future_data(df, future_date)
        
        if prediction_tool != "Ensemble":
            future_df['predicted_trips'] = best_tool_trip.predict(X_future)
        else:
            xgb_future_trip = xgb_trip.predict(X_future)
            rf_future_trip = rf_trip.predict(X_future)
            gbr_future_trip = gbr_trip.predict(X_future)
            future_df['predicted_trips'] = combine_predictions(xgb_future_trip, rf_future_trip, gbr_future_trip, trip_weights)
        
        display_df = future_df[['predicted_trips', 'date', 'day_name']].reset_index(drop=True)
        display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['Day'] = display_df['day_name']
        display_df['Predicted Trips'] = display_df['predicted_trips'].round(0).astype(int)
        display_df = display_df[['Date', 'Day', 'Predicted Trips']]
        
        st.markdown("### 📅 Weekly Trip Forecast")
        st.dataframe(display_df, use_container_width=True)
        
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df['date'], y=future_df['predicted_trips'], name='Predicted Trips', line=dict(color='#EF553B')))
        fig_future.update_layout(title=f'Predicted Trips for Week of {future_date.strftime("%B %d, %Y")}', xaxis_title='Date', yaxis_title='Number of Trips')
        st.plotly_chart(fig_future, use_container_width=True)
        
        peak_future_day = future_df.loc[future_df['predicted_trips'].idxmax()]
        quiet_future_day = future_df.loc[future_df['predicted_trips'].idxmin()]
        
        st.markdown(f"""
        ### 🚀 Key Insights
        - **Busiest Day**: {peak_future_day['day_name']} ({peak_future_day['date'].strftime('%Y-%m-%d')}) with ~{peak_future_day['predicted_trips']:.0f} trips! 🏙️
        - **Quietest Day**: {quiet_future_day['day_name']} ({quiet_future_day['date'].strftime('%Y-%m-%d')}) with ~{quiet_future_day['predicted_trips']:.0f} trips. 😴
        - **Action Plan**:
          - 📋 Schedule extra drivers on {peak_future_day['day_name']} to handle high demand! 🚕
          - 🔧 Use {quiet_future_day['day_name']} for vehicle maintenance or driver training. 🛠️
          - 🎯 Monitor holidays or events that could spike trips further! 🎉
        """)
        
        if prediction_tool in ["Random Forest", "XGBoost", "Gradient Boosting"]:
            st.subheader("🔍 What Drives Predictions?")
            st.markdown("Top factors shaping our trip forecasts! 🧠")
            best_tool = train_prediction_tool(X_train_trip, y_train_trip, prediction_tool).best_estimator_
            factor_names = [f'{factor}_past_day_{i+1}' for i in range(14) for factor in ['Trips', 'Active_Vehicles', 'Trips_Lag1', 'Trips_Lag7', 'Vehicles_Lag1', 
                                                                                        'Rolling_Avg_Trips_7d', 'Rolling_Avg_Vehicles_7d', 'Dayofweek', 'Weekend', 'Holiday']]
            factor_importance = pd.DataFrame({
                'Factor': factor_names[:len(best_tool.feature_importances_)],
                'Importance': best_tool.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_factors = px.bar(factor_importance.head(10), x='Importance', y='Factor', title='🏆 Top 10 Prediction Drivers',
                                color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_factors, use_container_width=True)
            
            top_factor = factor_importance.iloc[0]['Factor']
            st.markdown(f"**Top Driver**: {top_factor}. This factor heavily influences trip predictions—keep it in mind for planning! 📝")
else:
    st.warning("⚠️ Please enter a valid CSV file path to explore your data! 😞")