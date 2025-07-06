# Demand Forecasting for Inventory Optimization

## Live App

_ [[Live App](https://demandforecastinginventory.streamlit.app/)]_  
This project is an **interactive time-series forecasting web application** built with **Streamlit** and powered by **Prophet**, designed to help businesses forecast product demand across multiple product families, aiding in inventory planning and optimization.

---

## Objective

The primary goal of this project is to:
- Forecast product demand for the next 90 days using historical sales data.
- Help businesses manage inventory by understanding sales patterns and future projections.
- Provide interactive visualizations for data exploration, trends, and model performance.

---

## Problem Statement

Inventory overstocking leads to increased holding costs, while understocking results in missed sales. This application aims to provide **data-driven demand forecasting** to strike the right balance using historical data and advanced time-series forecasting.

---

## How It Works — Full Walkthrough

The app is divided into **four main sections** via sidebar navigation:

---

### Upload & Overview

- **Data Loading**: The app reads `train.csv`, assumes the file contains sales history with columns: `date`, `family`, and `sales`.
- **Preprocessing**:
  - Column names are sanitized (e.g., spaces replaced with underscores and lowercase).
  - The `date` column is converted to a `datetime` format.
  - Two new columns are created: `year` and `month` from the `date`.

- **Missing Value Analysis**:
  - Displays a table showing count and percentage of missing values for each column.
  - Useful for quickly assessing data quality.

---

### EDA & Insights

- **Aggregate Daily Sales**: Sales are grouped by `date` and `family`, then pivoted into a wide format (date as index, families as columns).
- **Interactive Filters**: You can select specific years and months from the sidebar to analyze.
- **Insights Provided**:
  - **Average Daily Sales**: Shows how much each product family sells on average.
  - **Total Sales Bar Chart**: Ranks families by total sales over the selected period.
  - **Line Plot**: Displays historical sales trend over time for a selected product family.
  - **Monthly Comparison**: If multiple years are selected, it compares the same months across different years to spot seasonality.

---

### Forecasting

- **Select Product Family**: Choose one family to forecast using Facebook Prophet.
- **Prepare Data for Prophet**:
  - Rename columns to `ds` (date) and `y` (value) as Prophet expects.
- **Modeling with Prophet**:
  - Fit a time-series model using historical data.
  - Forecast for the next **90 days**.

- **Forecast Visualization**:
  - Line chart showing:
    - Forecasted values (`yhat`)
    - Upper and lower confidence intervals
    - Actual sales (for overlap)
  - **Trend Plot**: Visualizes long-term direction of sales.
  - **Weekly Seasonality**: Shows how sales vary by day of the week.
  - **Yearly Seasonality**: Captures patterns that repeat annually.

- **Download Forecast Data**: Option to export forecast to a CSV file.

---

### Model Metrics

- **Model Evaluation**:
  - Performs **cross-validation** using Prophet’s built-in tools.
    - Initial: 730 days
    - Horizon: 90 days
    - Period: 180 days
  - Calculates metrics like:
    - MAPE (Mean Absolute Percentage Error)
    - RMSE
    - Coverage, etc.
- **Visualization**:
  - Plots MAPE across forecast horizons to judge prediction quality.

---

### Technologies Used

| Tool/Library        | Purpose                                    |
|---------------------|--------------------------------------------|
| **Streamlit**       | Web app frontend and interactivity         |
| **Facebook Prophet**| Forecasting engine for time series         |
| **Plotly**          | Interactive charts and graphs              |
| **Matplotlib**      | Used in model metric plots                 |
| **Pandas**          | Data manipulation                          |
| **NumPy**           | Numerical calculations                     |

---

## Input Dataset Format

`train.csv` must have the following structure:

| date       | family       | sales |
|------------|--------------|-------|
| 2020-01-01 | BEVERAGES    | 123   |
| 2020-01-01 | BREAD/BAKERY | 250   |

- `date`: Should be in ISO 8601 format (YYYY-MM-DD)
- `family`: Product category
- `sales`: Total sales units for that date and family

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app
```
### 2. Install Requirements
Create a virtual environment
```bash
pip install -r requirements.txt
```
### 3. Add Data File
Place train.csv in the root directory.

### 4. Launch App
```bash
streamlit run app.py
```

