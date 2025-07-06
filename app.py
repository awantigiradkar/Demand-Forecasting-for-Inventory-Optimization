import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Demand Forecasting for Inventory Optimization")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Overview", "EDA & Insights", "Forecasting", "Model Metrics"])

# Load pre-existing data
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    return df

# Missing data summary
def missing_data(input_data):
    total = input_data.isnull().sum()
    percent = (input_data.isnull().sum()/input_data.isnull().count()*100)
    table = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    table['Types'] = [str(input_data[col].dtype) for col in input_data.columns]
    return table

# Load the dataset
try:
    df = load_data()

    # Multi-select Month and Year filter
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    all_years = sorted(df['year'].unique(), reverse=True)
    all_months = pd.date_range(start='2020-01-01', periods=12, freq='MS').strftime('%B')

    selected_years = st.sidebar.multiselect("Select Year(s)", all_years, default=all_years[:1])
    selected_months = st.sidebar.multiselect("Select Month(s)", all_months, default=list(all_months))

    df = df[df['year'].isin(selected_years) & df['month'].isin(selected_months)]

    agg_df = df.groupby(['date', 'family'])['sales'].sum().reset_index()
    total_sales_df = agg_df.pivot(index='date', columns='family', values='sales')

    if page == "Upload & Overview":
        st.markdown("""
        This app uses a preloaded dataset to forecast future demand using Facebook Prophet.
        Explore data quality, trends, and Prophet's prediction performance.
        """)

        st.subheader("Raw Data")
        st.dataframe(df.head())

        st.subheader("Missing Data Summary")
        missing = missing_data(df)
        st.dataframe(missing)

    elif page == "EDA & Insights":
        st.subheader("Sales Summary Insights")
        avg_daily_sales = total_sales_df.mean()
        st.write("🔹 Average Daily Sales per Family")
        st.dataframe(avg_daily_sales.sort_values(ascending=False).round(2))

        st.write("Total Sales by Family")
        total_sales = total_sales_df.sum().sort_values(ascending=False)
        st.bar_chart(total_sales)

        selected_family = st.selectbox("Select Product Family to Visualize", total_sales_df.columns)

        st.subheader(f"Historical Sales - {selected_family}")
        ts_df = total_sales_df[[selected_family]].reset_index()
        fig = px.line(ts_df, x='date', y=selected_family, color_discrete_sequence=['indigo'], title=f"Sales Over Time for {selected_family}")
        fig.update_layout(xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig)

        if len(selected_years) > 1:
            st.subheader("Side-by-Side Yearly Comparison")
            ts_df['year'] = ts_df['date'].dt.year
            ts_df['month'] = ts_df['date'].dt.strftime('%b')
            fig_comp = px.line(ts_df, x='month', y=selected_family, color='year', markers=True,
                               title=f"{selected_family} Sales by Month Across Years")
            st.plotly_chart(fig_comp)

    elif page == "Forecasting":
        selected_family = st.selectbox("Select Product Family for Forecasting", total_sales_df.columns)
        st.subheader("Forecast Future Sales")
        forecast_df = total_sales_df[selected_family].reset_index()
        forecast_df.columns = ['ds', 'y']

        m = Prophet()
        m.fit(forecast_df)

        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)

        # Merge actuals with forecast
        merged = pd.merge(forecast, forecast_df, on='ds', how='left')

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], name='Forecast'))
        fig1.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
        fig1.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
        fig1.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], name='Actual Sales', line=dict(color='green')))
        fig1.update_layout(title="Forecast with Actual Sales", xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig1)

        st.subheader("Forecast Components (Interactive)")
        if 'trend' in forecast.columns:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend', line=dict(color='blue')))
            fig_trend.update_layout(title="Trend Component", xaxis_title="Date", yaxis_title="Trend")
            st.plotly_chart(fig_trend)

        if 'weekly' in forecast.columns:
            weekly_df = forecast[['ds', 'weekly']].copy()
            weekly_df['day'] = weekly_df['ds'].dt.day_name()
            weekly_avg = weekly_df.groupby('day').mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()

            fig_weekly = px.bar(weekly_avg, x='day', y='weekly', color='weekly', title="Weekly Seasonality Component",
                                labels={'weekly': 'Effect', 'day': 'Day of Week'}, color_continuous_scale='Blues')
            st.plotly_chart(fig_weekly)

        if 'yearly' in forecast.columns:
            fig_yearly = go.Figure()
            fig_yearly.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Yearly Seasonality', line=dict(color='purple')))
            fig_yearly.update_layout(title="Yearly Seasonality Component", xaxis_title="Date", yaxis_title="Effect")
            st.plotly_chart(fig_yearly)

        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast Data", data=csv, file_name="forecast.csv", mime='text/csv')

    elif page == "Model Metrics":
        selected_family = st.selectbox("Select Product Family for Metrics", total_sales_df.columns)
        st.subheader("Forecast Cross-Validation Metrics")
        forecast_df = total_sales_df[selected_family].reset_index()
        forecast_df.columns = ['ds', 'y']

        m = Prophet()
        m.fit(forecast_df)

        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='90 days')
        df_p = performance_metrics(df_cv)
        st.dataframe(df_p)

        fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
        st.pyplot(fig_cv)

except Exception as e:
    st.error(f"An error occurred: {e}. Make sure 'data.csv' is present in the directory.")
