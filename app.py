import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import uuid
import io
from PIL import Image
import plotly.io as pio

# Check for statsmodels
try:
    import statsmodels
    trendline_available = True
except ImportError:
    trendline_available = False

st.set_page_config(layout="wide")
st.title("Dynamic Data Visualization Dashboard")
st.caption("Developed by Abhinay")

# Initialize session state
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'df_dashboard' not in st.session_state:
    st.session_state.df_dashboard = None
if 'default_visualizations' not in st.session_state:
    st.session_state.default_visualizations = []
if 'custom_visualizations' not in st.session_state:
    st.session_state.custom_visualizations = []
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {'viz': {}, 'dashboard': {}, 'filtered_view': {}}
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None
if 'selected_filter_mode' not in st.session_state:
    st.session_state.selected_filter_mode = "Global Dashboard Filters"
if 'temp_viz' not in st.session_state:
    st.session_state.temp_viz = None
if 'dashboard_mode' not in st.session_state:
    st.session_state.dashboard_mode = "Default Dashboard"
if 'show_dataset' not in st.session_state:
    st.session_state.show_dataset = False

@st.cache_data
def load_data(file):
    try:
        file.seek(0)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
        if df.empty or df.columns.size == 0:
            raise ValueError("The uploaded file is empty or has no columns.")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {str(e)}.")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

@st.cache_data
def clean_data(df, fill_method='mean', custom_fill_value=None, interpolate_method=None, 
               drop_duplicates=True, remove_outliers=False, drop_high_null_rows=False, 
               null_threshold=0.5, replace_values=None, sample_size=None):
    df_cleaned = df.copy()
    
    with st.spinner("Handling missing values..."):
        if fill_method == 'custom' and custom_fill_value is not None:
            if isinstance(custom_fill_value, (int, float, str)):
                df_cleaned = df_cleaned.fillna(custom_fill_value)
            else:
                st.warning("Custom fill value must be a number or string.")
        elif fill_method == 'ffill':
            df_cleaned = df_cleaned.ffill()
        elif fill_method == 'bfill':
            df_cleaned = df_cleaned.bfill()
        elif fill_method == 'interpolate' and interpolate_method:
            df_cleaned = df_cleaned.interpolate(method=interpolate_method)
        elif fill_method == 'mean':
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
        elif fill_method == 'median':
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
        elif fill_method == 'mode':
            categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                mode = df_cleaned[col].mode()
                if not mode.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode[0])
        elif fill_method == 'drop':
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            if len(df_cleaned) < initial_rows:
                st.warning(f"Dropped {initial_rows - len(df_cleaned)} rows with missing values.")

    if drop_high_null_rows:
        initial_rows = len(df_cleaned)
        threshold = len(df_cleaned.columns) * null_threshold
        df_cleaned = df_cleaned.dropna(thresh=threshold)
        if len(df_cleaned) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(df_cleaned)} rows with more than {int(null_threshold*100)}% null values.")

    if drop_duplicates:
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        if len(df_cleaned) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(df_cleaned)} duplicate rows.")

    if remove_outliers:
        numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            if len(df_cleaned) < initial_rows:
                st.warning(f"Removed {initial_rows - len(df_cleaned)} rows with outliers in {col}.")

    if replace_values:
        for old_value, new_value in replace_values.items():
            df_cleaned = df_cleaned.replace(old_value, new_value)
            st.warning(f"Replaced '{old_value}' with '{new_value}'.")

    if sample_size and len(df_cleaned) > sample_size:
        st.warning(f"Dataset is large. Sampling to {sample_size} rows for performance.")
        df_cleaned = df_cleaned.sample(n=sample_size, random_state=42)

    return df_cleaned

def apply_filters(df, categorical_filters, numeric_filters, date_filter, custom_query):
    df_filtered = df.copy()
    
    for col, selected_vals in categorical_filters.items():
        if selected_vals:
            df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
    
    for col, (min_val, max_val) in numeric_filters.items():
        if min_val is not None and max_val is not None:
            df_filtered = df_filtered[(df_filtered[col] >= min_val) & (df_filtered[col] <= max_val)]
    
    if date_filter.get('use_date_filter') and date_filter.get('date_col'):
        date_col = date_filter['date_col']
        date_format = date_filter.get('date_format')
        start_date = date_filter.get('start_date')
        end_date = date_filter.get('end_date')
        try:
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], format=date_format if date_format else None, errors='coerce')
            if df_filtered[date_col].isna().all():
                st.warning(f"Column '{date_col}' contains no valid dates.")
            else:
                df_filtered = df_filtered[(df_filtered[date_col].dt.date >= start_date) & (df_filtered[date_col].dt.date <= end_date)]
        except Exception as e:
            st.warning(f"Could not parse '{date_col}' as datetime: {str(e)}")
    
    if custom_query:
        try:
            df_filtered = df_filtered.query(custom_query, engine='python')
        except Exception as e:
            st.error(f"Invalid query: {str(e)}")
    
    return df_filtered

def get_suitable_chart_types(numeric_columns, categorical_columns):
    chart_types = []
    if numeric_columns:
        chart_types.extend(["Scatter Plot", "Histogram"])
        if len(numeric_columns) >= 2:
            chart_types.append("Heatmap")
    if categorical_columns:
        chart_types.extend(["Pie Chart", "Bar Chart"])
    if numeric_columns and categorical_columns:
        chart_types.extend(["Box Plot", "Violin Plot"])
    if numeric_columns and len(chart_types) > 0:
        chart_types.append("Line Chart")
    return chart_types if chart_types else ["Histogram"]

def generate_default_visualizations(df, numeric_columns, categorical_columns):
    visualizations = []
    suitable_charts = get_suitable_chart_types(numeric_columns, categorical_columns)
    
    if "Bar Chart" in suitable_charts and categorical_columns and numeric_columns:
        viz = {
            'id': str(uuid.uuid4()),
            'chart_type': 'Bar Chart',
            'x_axis': categorical_columns[0],
            'y_axis': numeric_columns[0],
            'agg_method': 'sum',
            'show_trendline': False,
            'categorical_filters': {},
            'numeric_filters': {},
            'date_filter': {},
            'custom_query': ''
        }
        visualizations.append(viz)
    
    if "Pie Chart" in suitable_charts and categorical_columns:
        viz = {
            'id': str(uuid.uuid4()),
            'chart_type': 'Pie Chart',
            'x_axis': categorical_columns[0],
            'y_axis': numeric_columns[0] if numeric_columns else None,
            'agg_method': 'count' if not numeric_columns else 'sum',
            'show_trendline': False,
            'categorical_filters': {},
            'numeric_filters': {},
            'date_filter': {},
            'custom_query': ''
        }
        visualizations.append(viz)
    
    if "Scatter Plot" in suitable_charts and len(numeric_columns) >= 2:
        viz = {
            'id': str(uuid.uuid4()),
            'chart_type': 'Scatter Plot',
            'x_axis': numeric_columns[0],
            'y_axis': numeric_columns[1],
            'agg_method': None,
            'show_trendline': trendline_available,
            'categorical_filters': {},
            'numeric_filters': {},
            'date_filter': {},
            'custom_query': ''
        }
        visualizations.append(viz)
    
    if "Heatmap" in suitable_charts and len(numeric_columns) >= 2:
        viz = {
            'id': str(uuid.uuid4()),
            'chart_type': 'Heatmap',
            'x_axis': None,
            'y_axis': None,
            'agg_method': None,
            'show_trendline': False,
            'categorical_filters': {},
            'numeric_filters': {},
            'date_filter': {},
            'custom_query': ''
        }
        visualizations.append(viz)
    
    if "Histogram" in suitable_charts and numeric_columns:
        viz = {
            'id': str(uuid.uuid4()),
            'chart_type': 'Histogram',
            'x_axis': numeric_columns[0],
            'y_axis': None,
            'agg_method': None,
            'show_trendline': False,
            'categorical_filters': {},
            'numeric_filters': {},
            'date_filter': {},
            'custom_query': ''
        }
        visualizations.append(viz)
    
    return visualizations

# File Upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader")

# Handle file change
def get_file_id(file):
    if file is None:
        return None
    return getattr(file, "name", None) or str(file)

current_file_id = get_file_id(uploaded_file)
previous_file_id = st.session_state.get("last_uploaded_file_id", None)

if uploaded_file and (current_file_id != previous_file_id):
    st.session_state.last_uploaded_file_id = current_file_id
    st.cache_data.clear()
    try:
        original_df = load_data(uploaded_file)
        st.session_state.original_df = original_df.copy()
        st.session_state.df_dashboard = original_df.copy()
        st.session_state.filtered_df = original_df.copy()
        numeric_columns = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = original_df.select_dtypes(include=['object']).columns.tolist()
        st.session_state.default_visualizations = generate_default_visualizations(original_df, numeric_columns, categorical_columns)
        st.session_state.custom_visualizations = []
        st.session_state.active_filters = {'viz': {}, 'dashboard': {}, 'filtered_view': {}}
        st.session_state.selected_filter_mode = "Global Dashboard Filters"
        st.session_state.temp_viz = None
        st.session_state.dashboard_mode = "Default Dashboard"
        st.session_state.show_dataset = False
    except Exception as e:
        st.error(str(e))
        st.stop()

# Main app logic
if uploaded_file:
    try:
        original_df = load_data(uploaded_file)
        if 'original_df' not in st.session_state:
            st.session_state.original_df = original_df.copy()
        df = st.session_state.original_df.copy()
        if 'df_dashboard' not in st.session_state:
            st.session_state.df_dashboard = df.copy()
        if 'filtered_df' not in st.session_state:
            st.session_state.filtered_df = df.copy()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Define columns
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    date_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dataset View", "🧹 Clean Data", "💻 Filters & Visualization", "📈 Dashboard", "🔍 Insights & Filtered Data"])

    with tab1:
        st.subheader("📋 Dataset View")
        st.write("View the original dataset after cleaning (no filters applied).")
        if st.session_state.original_df.empty:
            st.warning("The dataset is empty.")
        else:
            st.write(f"Dataset: {st.session_state.original_df.shape[0]} rows, {st.session_state.original_df.shape[1]} columns")
                        # ...existing code...
            if st.button("Toggle Dataset Display", key="toggle_dataset"):
                st.session_state.show_dataset = not st.session_state.show_dataset
            
            if st.session_state.show_dataset:
                try:
                    df_to_show = st.session_state.original_df
                    # Check for empty DataFrame or no columns
                    if df_to_show is not None and not df_to_show.empty and len(df_to_show.columns) > 0:
                        st.dataframe(df_to_show.head(100), use_container_width=True)
                        if df_to_show.shape[0] > 100:
                            st.info("Only first 100 rows are shown for performance.")
                    elif df_to_show is not None and len(df_to_show.columns) == 0:
                        st.warning("The dataset has no columns.")
                    else:
                        st.warning("The dataset is empty.")
                except Exception as e:
                    st.error(f"Error displaying dataset: {str(e)}")
            # ...existing code...
 
            csv = st.session_state.original_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Original Dataset as CSV",
                data=csv,
                file_name="original_dataset.csv",
                mime="text/csv",
                key="download_original"
            )

    with tab2:
        st.subheader("🧹 Clean Data")
        use_sampling = st.checkbox("Sample Large Datasets", value=len(df) > 10000, key="sample_large")
        sample_size = st.slider("Sample Size (rows)", 1000, min(100000, len(df)), 10000, step=1000, key="sample_size") if use_sampling else None

        with st.expander("Data Cleaning Options", expanded=True):
            fill_method = st.selectbox("Handle Missing Values", ["mean", "median", "mode", "custom", "ffill", "bfill", "interpolate", "drop"], key="fill_method")
            if fill_method == "custom":
                custom_fill_value = st.text_input("Enter custom fill value (number or string)", key="custom_fill")
                try:
                    custom_fill_value = float(custom_fill_value) if custom_fill_value.replace('.','',1).isdigit() else custom_fill_value
                except ValueError:
                    custom_fill_value = custom_fill_value
            else:
                custom_fill_value = None
            if fill_method == "interpolate":
                interpolate_method = st.selectbox("Interpolation Method", ["linear", "quadratic", "cubic"], key="interpolate_method")
            else:
                interpolate_method = None
            drop_duplicates = st.checkbox("Drop Duplicate Rows", value=True, key="drop_duplicates")
            remove_outliers = st.checkbox("Remove Outliers (IQR Method)", value=False, key="remove_outliers")
            drop_high_null_rows = st.checkbox("Drop Rows with High Null Percentage", value=False, key="drop_high_null")
            null_threshold = st.slider("Null Threshold (%)", 0.0, 1.0, 0.5, 0.1, key="null_threshold") if drop_high_null_rows else 0.5
            replace_values = {}
            if st.checkbox("Replace Specific Values", key="replace_values"):
                cols = df.columns.tolist()
                col_to_replace = st.selectbox("Select Column to Replace Values", cols, key="col_to_replace")
                old_value = st.text_input("Old Value to Replace", key="old_value")
                new_value = st.text_input("New Value", key="new_value")
                if old_value and new_value:
                    replace_values[old_value] = new_value
            if st.button("Apply Cleaning", key="apply_cleaning"):
                with st.spinner("Cleaning data..."):
                    cleaned_df = clean_data(st.session_state.original_df, fill_method, custom_fill_value, interpolate_method, 
                                            drop_duplicates, remove_outliers, drop_high_null_rows, null_threshold, replace_values, sample_size)
                    
                    st.session_state.df_dashboard = cleaned_df.copy()
                    st.session_state.filtered_df = cleaned_df.copy()
                    st.session_state.default_visualizations = generate_default_visualizations(cleaned_df, 
                                                                                           cleaned_df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                                                                                           cleaned_df.select_dtypes(include=['object']).columns.tolist())
                    st.session_state.custom_visualizations = []
                    st.session_state.active_filters = {'viz': {}, 'dashboard': {}, 'filtered_view': {}}
                    st.session_state.selected_filter_mode = "Global Dashboard Filters"
                    st.session_state.temp_viz = None
                    st.session_state.dashboard_mode = "Default Dashboard"
                    st.session_state.show_dataset = False
                    st.success("Data cleaned successfully!")

    with tab3:
        st.subheader("💻 Filters & Visualization")
        st.write("Create visualizations with specific filters and add them to the Custom Dashboard.")
        
        df_viz = df.copy()
        if df_viz.empty or not columns:
            st.warning("No data available after cleaning.")
            st.stop()

        # Initialize temporary visualization
        if st.session_state.temp_viz is None:
            suitable_charts = get_suitable_chart_types(numeric_columns, categorical_columns)
            default_chart = suitable_charts[0] if suitable_charts else "Histogram"
            x_axis = categorical_columns[0] if categorical_columns and default_chart in ["Bar Chart", "Pie Chart", "Box Plot", "Violin Plot"] else numeric_columns[0] if numeric_columns else None
            y_axis = numeric_columns[0] if numeric_columns and default_chart not in ["Pie Chart", "Histogram"] else None
            if default_chart == "Scatter Payroll" and len(numeric_columns) >= 2:
                y_axis = numeric_columns[1]
            st.session_state.temp_viz = {
                'chart_type': default_chart,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'agg_method': 'sum' if default_chart == "Pie Chart" and numeric_columns else 'count' if default_chart == "Pie Chart" else None,
                'show_trendline': False,
                'categorical_filters': {col: [] for col in categorical_columns},
                'numeric_filters': {col: (float(df[col].min()), float(df[col].max())) for col in numeric_columns if df[col].min() != df[col].max()},
                'date_filter': {'date_col': date_columns[0] if date_columns else None, 'date_format': None, 'use_date_filter': False, 'start_date': None, 'end_date': None},
                'custom_query': ''
            }

        with st.sidebar.expander("Visualization Options", expanded=True):
            suitable_charts = get_suitable_chart_types(numeric_columns, categorical_columns)

            def update_chart_type():
                st.session_state.temp_viz['chart_type'] = st.session_state.viz_chart_type
                chart_type = st.session_state.viz_chart_type
                if chart_type in ["Bar Chart", "Box Plot", "Violin Plot"]:
                    st.session_state.temp_viz['x_axis'] = categorical_columns[0] if categorical_columns else None
                    st.session_state.temp_viz['y_axis'] = numeric_columns[0] if numeric_columns else None
                elif chart_type in ["Line Chart", "Scatter Plot"]:
                    st.session_state.temp_viz['x_axis'] = numeric_columns[0] if numeric_columns else None
                    st.session_state.temp_viz['y_axis'] = numeric_columns[1] if len(numeric_columns) >= 2 else numeric_columns[0] if numeric_columns else None
                elif chart_type == "Pie Chart":
                    st.session_state.temp_viz['x_axis'] = categorical_columns[0] if categorical_columns else None
                    st.session_state.temp_viz['y_axis'] = numeric_columns[0] if numeric_columns else None
                elif chart_type == "Histogram":
                    st.session_state.temp_viz['x_axis'] = numeric_columns[0] if numeric_columns else None
                    st.session_state.temp_viz['y_axis'] = None
                elif chart_type == "Heatmap":
                    st.session_state.temp_viz['x_axis'] = None
                    st.session_state.temp_viz['y_axis'] = None
                st.session_state.temp_viz['agg_method'] = 'sum' if chart_type == "Pie Chart" and numeric_columns else 'count' if chart_type == "Pie Chart" else None
                st.session_state.temp_viz['show_trendline'] = False

            chart_type = st.radio(
                "Chart Type",
                suitable_charts,
                index=suitable_charts.index(st.session_state.temp_viz['chart_type']) if st.session_state.temp_viz['chart_type'] in suitable_charts else 0,
                key="viz_chart_type",
                on_change=update_chart_type
            )
            
            axis_options = categorical_columns if chart_type in ["Bar Chart", "Pie Chart", "Box Plot", "Violin Plot"] else columns
            if chart_type != "Heatmap":
                if axis_options:
                    st.session_state.temp_viz['x_axis'] = st.selectbox(
                        "X-axis",
                        axis_options,
                        index=axis_options.index(st.session_state.temp_viz['x_axis']) if st.session_state.temp_viz['x_axis'] in axis_options else 0,
                        key="viz_x_axis"
                    )
                else:
                    st.warning("No suitable columns for X-axis.")
                    st.stop()
                
                if chart_type not in ["Pie Chart", "Histogram"]:
                    if numeric_columns:
                        st.session_state.temp_viz['y_axis'] = st.selectbox(
                            "Y-axis (numeric)",
                            numeric_columns,
                            index=numeric_columns.index(st.session_state.temp_viz['y_axis']) if st.session_state.temp_viz['y_axis'] in numeric_columns else 0,
                            key="viz_y_axis"
                        )
                    else:
                        st.warning("No numeric columns available for Y-axis.")
                        st.stop()
            
            if chart_type == "Pie Chart":
                st.session_state.temp_viz['agg_method'] = st.selectbox(
                    "Aggregation Method",
                    ["sum", "mean", "count"],
                    index=["sum", "mean", "count"].index(st.session_state.temp_viz['agg_method']) if st.session_state.temp_viz['agg_method'] else 0,
                    key="viz_agg_method"
                )
            
            if chart_type == "Scatter Plot":
                st.session_state.temp_viz['show_trendline'] = st.checkbox(
                    "Show Trendline",
                    value=st.session_state.temp_viz['show_trendline'],
                    disabled=not trendline_available,
                    key="viz_trendline"
                )

        with st.sidebar.expander("Visualization Filters", expanded=True):
            for col in categorical_columns:
                st.session_state.temp_viz['categorical_filters'][col] = st.multiselect(
                    f"Filter by {col}",
                    options=df_viz[col].unique(),
                    default=st.session_state.temp_viz['categorical_filters'].get(col, []),
                    key=f"viz_cat_{col}"
                )
                if st.session_state.temp_viz['categorical_filters'][col]:
                    st.session_state.active_filters['viz'][f"{col}_categorical"] = st.session_state.temp_viz['categorical_filters'][col]
            
            for col in numeric_columns:
                min_val, max_val = float(df_viz[col].min()), float(df_viz[col].max())
                if min_val != max_val:
                    st.session_state.temp_viz['numeric_filters'][col] = st.slider(
                        f"Range for {col}",
                        min_val, max_val,
                        st.session_state.temp_viz['numeric_filters'].get(col, (min_val, max_val)),
                        key=f"viz_num_{col}"
                    )
                    if st.session_state.temp_viz['numeric_filters'][col] != (min_val, max_val):
                        st.session_state.active_filters['viz'][f"{col}_numeric"] = st.session_state.temp_viz['numeric_filters'][col]
                else:
                    st.write(f"{col}: Constant value {min_val}")
                    st.session_state.temp_viz['numeric_filters'][col] = (min_val, max_val)

            if date_columns:
                st.session_state.temp_viz['date_filter']['date_col'] = st.selectbox(
                    "Select Date Column",
                    date_columns,
                    index=date_columns.index(st.session_state.temp_viz['date_filter']['date_col']) if st.session_state.temp_viz['date_filter']['date_col'] in date_columns else 0,
                    key="viz_date_col"
                )
                st.session_state.temp_viz['date_filter']['date_format'] = st.text_input(
                    "Specify Date Format (e.g., %Y-%m-%d)",
                    value=st.session_state.temp_viz['date_filter']['date_format'],
                    key="viz_date_format"
                )
                st.session_state.temp_viz['date_filter']['use_date_filter'] = st.checkbox(
                    "Filter by Date",
                    value=st.session_state.temp_viz['date_filter']['use_date_filter'],
                    key="viz_date_filter"
                )
                if st.session_state.temp_viz['date_filter']['use_date_filter']:
                    st.session_state.temp_viz['date_filter']['start_date'] = st.date_input(
                        "Start Date",
                        value=st.session_state.temp_viz['date_filter']['start_date'] or df_viz[st.session_state.temp_viz['date_filter']['date_col']].min().date(),
                        key="viz_start_date"
                    )
                    st.session_state.temp_viz['date_filter']['end_date'] = st.date_input(
                        "End Date",
                        value=st.session_state.temp_viz['date_filter']['end_date'] or df_viz[st.session_state.temp_viz['date_filter']['date_col']].max().date(),
                        key="viz_end_date"
                    )
                    st.session_state.active_filters['viz']['date_filter'] = f"{st.session_state.temp_viz['date_filter']['date_col']} from {st.session_state.temp_viz['date_filter']['start_date']} to {st.session_state.temp_viz['date_filter']['end_date']}"

            st.session_state.temp_viz['custom_query'] = st.text_area(
                "Enter custom query (e.g., Value > 100)",
                value=st.session_state.temp_viz['custom_query'],
                height=100,
                key="viz_query"
            )
            if st.session_state.temp_viz['custom_query']:
                st.session_state.active_filters['viz']['custom_query'] = st.session_state.temp_viz['custom_query']

            if st.button("Preview Filters", key="preview_viz_filters"):
                df_preview = apply_filters(
                    df.copy(),
                    st.session_state.temp_viz['categorical_filters'],
                    st.session_state.temp_viz['numeric_filters'],
                    st.session_state.temp_viz['date_filter'],
                    st.session_state.temp_viz['custom_query']
                )
                if df_preview.empty:
                    st.warning("The applied filters result in an empty dataset. Please adjust your filters or query.")
                else:
                    st.dataframe(df_preview.head(100), use_container_width=True)

        # Apply filters and render visualization
        df_viz = apply_filters(
            df.copy(),
            st.session_state.temp_viz['categorical_filters'],
            st.session_state.temp_viz['numeric_filters'],
            st.session_state.temp_viz['date_filter'],
            st.session_state.temp_viz['custom_query']
        )

        st.write("### Preview of Filtered Data")
        if df_viz.empty:
            st.warning("The applied filters or query result in an empty dataset. Please adjust your filters or query.")
        else:
            st.dataframe(df_viz.head(100), use_container_width=True)

        if (st.session_state.temp_viz['chart_type'] == "Heatmap" and len(numeric_columns) >= 2) or \
           (st.session_state.temp_viz['x_axis'] and (st.session_state.temp_viz['y_axis'] or st.session_state.temp_viz['chart_type'] in ["Pie Chart", "Histogram"])):
            if df_viz.empty:
                st.warning("Cannot generate visualization: The filtered dataset is empty due to the applied filters or query.")
            else:
                try:
                    if st.session_state.temp_viz['chart_type'] == "Bar Chart":
                        fig = px.bar(df_viz, x=st.session_state.temp_viz['x_axis'], y=st.session_state.temp_viz['y_axis'], color=st.session_state.temp_viz['x_axis'], title=f"{st.session_state.temp_viz['y_axis']} by {st.session_state.temp_viz['x_axis']}")
                        fig.update_yaxes(range=[0, df_viz[st.session_state.temp_viz['y_axis']].max() * 1.1])
                    elif st.session_state.temp_viz['chart_type'] == "Line Chart":
                        fig = px.line(df_viz, x=st.session_state.temp_viz['x_axis'], y=st.session_state.temp_viz['y_axis'], title=f"{st.session_state.temp_viz['y_axis']} by {st.session_state.temp_viz['x_axis']}")
                        fig.update_yaxes(range=[0, df_viz[st.session_state.temp_viz['y_axis']].max() * 1.1])
                    elif st.session_state.temp_viz['chart_type'] == "Pie Chart":
                        if st.session_state.temp_viz['agg_method'] == "sum":
                            pie_data = df_viz.groupby(st.session_state.temp_viz['x_axis'])[st.session_state.temp_viz['y_axis']].sum().reset_index()
                        elif st.session_state.temp_viz['agg_method'] == "mean":
                            pie_data = df_viz.groupby(st.session_state.temp_viz['x_axis'])[st.session_state.temp_viz['y_axis']].mean().reset_index()
                        else:
                            pie_data = df_viz.groupby(st.session_state.temp_viz['x_axis']).size().reset_index(name='Count')
                        fig = px.pie(pie_data, values=pie_data.columns[1], names=st.session_state.temp_viz['x_axis'], title=f"Distribution of {st.session_state.temp_viz['x_axis']}")
                    elif st.session_state.temp_viz['chart_type'] == "Scatter Plot":
                        fig = px.scatter(df_viz, x=st.session_state.temp_viz['x_axis'], y=st.session_state.temp_viz['y_axis'], title=f"{st.session_state.temp_viz['y_axis']} vs {st.session_state.temp_viz['x_axis']}")
                        if st.session_state.temp_viz['show_trendline'] and trendline_available:
                            fig = px.scatter(df_viz, x=st.session_state.temp_viz['x_axis'], y=st.session_state.temp_viz['y_axis'], trendline="ols", title=f"{st.session_state.temp_viz['y_axis']} vs {st.session_state.temp_viz['x_axis']}")
                        fig.update_yaxes(range=[0, df_viz[st.session_state.temp_viz['y_axis']].max() * 1.1])
                    elif st.session_state.temp_viz['chart_type'] == "Box Plot":
                        fig = px.box(df_viz, x=st.session_state.temp_viz['x_axis'], y=st.session_state.temp_viz['y_axis'], color=st.session_state.temp_viz['x_axis'], title=f"{st.session_state.temp_viz['y_axis']} by {st.session_state.temp_viz['x_axis']}")
                        fig.update_yaxes(range=[0, df_viz[st.session_state.temp_viz['y_axis']].max() * 1.1])
                    elif st.session_state.temp_viz['chart_type'] == "Violin Plot":
                        fig = px.violin(df_viz, x=st.session_state.temp_viz['x_axis'], y=st.session_state.temp_viz['y_axis'], color=st.session_state.temp_viz['x_axis'], box=True, title=f"{st.session_state.temp_viz['y_axis']} by {st.session_state.temp_viz['x_axis']}")
                        fig.update_yaxes(range=[0, df_viz[st.session_state.temp_viz['y_axis']].max() * 1.1])
                    elif st.session_state.temp_viz['chart_type'] == "Histogram":
                        fig = px.histogram(df_viz, x=st.session_state.temp_viz['x_axis'], nbins=20, title=f"Histogram of {st.session_state.temp_viz['x_axis']}")
                    elif st.session_state.temp_viz['chart_type'] == "Heatmap":
                        corr = df_viz[numeric_columns].corr()
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                        st.pyplot(fig)
                    if st.session_state.temp_viz['chart_type'] != "Heatmap":
                        st.plotly_chart(fig, use_container_width=True, key=f"temp_viz_{st.session_state.temp_viz['chart_type']}_{st.session_state.temp_viz.get('x_axis','')}_{st.session_state.temp_viz.get('y_axis','')}")
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")

        if st.button("Add to Custom Dashboard", key="add_to_dashboard"):
            if (st.session_state.temp_viz['chart_type'] == "Heatmap" and len(numeric_columns) >= 2) or \
               (st.session_state.temp_viz['x_axis'] and (st.session_state.temp_viz['y_axis'] or st.session_state.temp_viz['chart_type'] in ["Pie Chart", "Histogram"])):
                if df_viz.empty:
                    st.warning("Cannot add visualization: The filtered dataset is empty due to the applied filters or query.")
                else:
                    st.session_state.custom_visualizations.append({
                        'id': str(uuid.uuid4()),
                        'chart_type': st.session_state.temp_viz['chart_type'],
                        'x_axis': st.session_state.temp_viz['x_axis'],
                        'y_axis': st.session_state.temp_viz['y_axis'],
                        'agg_method': st.session_state.temp_viz['agg_method'],
                        'show_trendline': st.session_state.temp_viz['show_trendline'],
                        'categorical_filters': st.session_state.temp_viz['categorical_filters'].copy(),
                        'numeric_filters': st.session_state.temp_viz['numeric_filters'].copy(),
                        'date_filter': st.session_state.temp_viz['date_filter'].copy(),
                        'custom_query': st.session_state.temp_viz['custom_query']
                    })
                    st.session_state.temp_viz = None  # Reset temp_viz
                    st.session_state.dashboard_mode = "Custom Dashboard"
                    st.success("Visualization added to Custom Dashboard with applied filters!")
                    st.rerun()
            else:
                st.warning("Please configure a valid visualization before adding to the dashboard.")

    with tab4:
        st.subheader("📈 Dashboard")
        st.write("View the Default or Custom Dashboard with global or per-visualization filters.")

        def update_dashboard_mode():
            st.session_state.dashboard_mode = st.session_state.dashboard_mode_selector

        dashboard_mode = st.radio(
            "Dashboard Mode",
            ["Default Dashboard", "Custom Dashboard"],
            index=0 if st.session_state.dashboard_mode == "Default Dashboard" else 1,
            key="dashboard_mode_selector",
            on_change=update_dashboard_mode
        )

        visualizations = st.session_state.default_visualizations if st.session_state.dashboard_mode == "Default Dashboard" else st.session_state.custom_visualizations

        # Debug line removed for production
        # st.write(f"Debug: Displaying {st.session_state.dashboard_mode} with {len(visualizations)} visualizations")

        filter_mode = st.radio(
            "Filter Mode",
            ["Global Dashboard Filters", "Per-Visualization Filters"],
            index=1,  # 0 for Global, 1 for Per-Visualization (default)
            key="filter_mode"
            )
        st.session_state.selected_filter_mode = filter_mode
        
        
        if filter_mode == "Global Dashboard Filters":
            with st.sidebar.expander("Global Dashboard Filters", expanded=True):
                dashboard_categorical_filters = {}
                for col in categorical_columns:
                    selected_vals = st.multiselect(f"Filter by {col}", options=df[col].unique(), key=f"dashboard_global_cat_{col}")
                    dashboard_categorical_filters[col] = selected_vals
                    if selected_vals:
                        st.session_state.active_filters['dashboard'][f"{col}_categorical"] = selected_vals
                
                dashboard_numeric_filters = {}
                for col in numeric_columns:
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    if min_val != max_val:
                        selected_range = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val), key=f"dashboard_global_num_{col}")
                        dashboard_numeric_filters[col] = selected_range
                        if selected_range != (min_val, max_val):
                            st.session_state.active_filters['dashboard'][f"{col}_numeric"] = selected_range
                    else:
                        st.write(f"{col}: Constant value {min_val}")
                        dashboard_numeric_filters[col] = (min_val, max_val)

                dashboard_date_filter = {}
                if date_columns:
                    dashboard_date_filter['date_col'] = st.selectbox("Select Date Column", date_columns, key="dashboard_global_date_col")
                    dashboard_date_filter['date_format'] = st.text_input("Specify Date Format (e.g., %Y-%m-%d)", value=None, key="dashboard_global_date_format")
                    dashboard_date_filter['use_date_filter'] = st.checkbox("Filter by Date", key="dashboard_global_date_filter")
                    if dashboard_date_filter['use_date_filter']:
                        dashboard_date_filter['start_date'] = st.date_input("Start Date", df[dashboard_date_filter['date_col']].min().date(), key="dashboard_global_start_date")
                        dashboard_date_filter['end_date'] = st.date_input("End Date", df[dashboard_date_filter['date_col']].max().date(), key="dashboard_global_end_date")
                        st.session_state.active_filters['dashboard']['date_filter'] = f"{dashboard_date_filter['date_col']} from {dashboard_date_filter['start_date']} to {dashboard_date_filter['end_date']}"

                dashboard_query = st.text_area("Enter custom query (e.g., Value > 100)", height=100, key="dashboard_global_query")
                if dashboard_query:
                    st.session_state.active_filters['dashboard']['custom_query'] = dashboard_query

                if st.button("Preview Global Filters", key="preview_global_filters"):
                    df_preview = apply_filters(df.copy(), dashboard_categorical_filters, dashboard_numeric_filters, dashboard_date_filter, dashboard_query)
                    if df_preview.empty:
                        st.warning("The applied global filters result in an empty dataset. Please adjust your filters or query.")
                    else:
                        st.dataframe(df_preview.head(100), use_container_width=True)

                if st.button("Apply Global Filters", key="apply_global_filters"):
                    st.session_state.df_dashboard = apply_filters(df.copy(), dashboard_categorical_filters, dashboard_numeric_filters, dashboard_date_filter, dashboard_query)
                    if st.session_state.df_dashboard.empty:
                        st.warning("Global filters result in an empty dataset. Consider adjusting your filters or query.")
                    else:
                        st.success("Global dashboard filters applied successfully!")

        if st.button("Reset Dashboard Filters", key="reset_dashboard_filters"):
            st.session_state.df_dashboard = df.copy()
            st.session_state.active_filters['dashboard'] = {}
            for viz_list in [st.session_state.default_visualizations, st.session_state.custom_visualizations]:
                for viz in viz_list:
                    viz['categorical_filters'] = {col: [] for col in categorical_columns}
                    viz['numeric_filters'] = {col: (float(df[col].min()), float(df[col].max())) for col in numeric_columns if df[col].min() != df[col].max()}
                    viz['date_filter'] = {'date_col': date_columns[0] if date_columns else None, 'date_format': None, 'use_date_filter': False, 'start_date': None, 'end_date': None}
                    viz['custom_query'] = ''
            st.success("Dashboard filters reset.")

        grid_columns = st.selectbox("Select Number of Columns for Grid Layout", [1, 2, 3], index=1, key="grid_columns")
        cols = st.columns(grid_columns)

        if not visualizations:
            st.info(f"No visualizations in the {st.session_state.dashboard_mode}. {'Add visualizations from the Filters & Visualization tab.' if st.session_state.dashboard_mode == 'Custom Dashboard' else 'Check your dataset for suitable columns.'}")
        else:
            for i, viz in enumerate(visualizations):
                with cols[i % grid_columns]:
                    st.write(f"#### Visualization {i+1}: {viz['chart_type']}")
                    df_viz = st.session_state.df_dashboard if filter_mode == "Global Dashboard Filters" else df.copy()
                    if filter_mode == "Per-Visualization Filters":
                        df_viz = apply_filters(df_viz, viz['categorical_filters'], viz['numeric_filters'], viz['date_filter'], viz['custom_query'])
                        st.session_state[f"filtered_data_{viz['id']}"] = df_viz
                        # Debug: Show applied filters
                        if viz['categorical_filters'] or viz['numeric_filters'] or viz['date_filter'].get('use_date_filter') or viz['custom_query']:
                            st.write("**Applied Filters:**")
                            if viz['categorical_filters']:
                                for col, vals in viz['categorical_filters'].items():
                                    if vals:
                                        st.write(f"- {col}: {vals}")
                            if viz['numeric_filters']:
                                for col, (min_val, max_val) in viz['numeric_filters'].items():
                                    if min_val != df[col].min() or max_val != df[col].max():
                                        st.write(f"- {col}: [{min_val}, {max_val}]")
                            if viz['date_filter'].get('use_date_filter'):
                                st.write(f"- {viz['date_filter']['date_col']}: from {viz['date_filter']['start_date']} to {viz['date_filter']['end_date']}")
                            if viz['custom_query']:
                                st.write(f"- Custom Query: {viz['custom_query']}")

                    if df_viz.empty:
                        st.warning(f"Visualization {i+1} cannot be displayed: The filtered dataset is empty due to the applied filters or query.")
                    else:
                        try:
                            if viz['chart_type'] == "Bar Chart":
                                fig = px.bar(df_viz, x=viz['x_axis'], y=viz['y_axis'], color=viz['x_axis'], title=f"{viz['y_axis']} by {viz['x_axis']}")
                                fig.update_yaxes(range=[0, df_viz[viz['y_axis']].max() * 1.1])
                            elif viz['chart_type'] == "Line Chart":
                                fig = px.line(df_viz, x=viz['x_axis'], y=viz['y_axis'], title=f"{viz['y_axis']} by {viz['x_axis']}")
                                fig.update_yaxes(range=[0, df_viz[viz['y_axis']].max() * 1.1])
                            elif viz['chart_type'] == "Pie Chart":
                                if viz['agg_method'] == "sum":
                                    pie_data = df_viz.groupby(viz['x_axis'])[viz['y_axis']].sum().reset_index()
                                elif viz['agg_method'] == "mean":
                                    pie_data = df_viz.groupby(viz['x_axis'])[viz['y_axis']].mean().reset_index()
                                else:
                                    pie_data = df_viz.groupby(viz['x_axis']).size().reset_index(name='Count')
                                fig = px.pie(pie_data, values=pie_data.columns[1], names=viz['x_axis'], title=f"Distribution of {viz['x_axis']}")
                            elif viz['chart_type'] == "Scatter Plot":
                                if viz['show_trendline'] and trendline_available:
                                    fig = px.scatter(df_viz, x=viz['x_axis'], y=viz['y_axis'], trendline="ols", title=f"{viz['y_axis']} vs {viz['x_axis']}")
                                else:
                                    fig = px.scatter(df_viz, x=viz['x_axis'], y=viz['y_axis'], title=f"{viz['y_axis']} vs {viz['x_axis']}")
                                fig.update_yaxes(range=[0, df_viz[viz['y_axis']].max() * 1.1])
                            elif viz['chart_type'] == "Box Plot":
                                fig = px.box(df_viz, x=viz['x_axis'], y=viz['y_axis'], color=viz['x_axis'], title=f"{viz['y_axis']} by {viz['x_axis']}")
                                fig.update_yaxes(range=[0, df_viz[viz['y_axis']].max() * 1.1])
                            elif viz['chart_type'] == "Violin Plot":
                                fig = px.violin(df_viz, x=viz['x_axis'], y=viz['y_axis'], color=viz['x_axis'], box=True, title=f"{viz['y_axis']} by {viz['x_axis']}")
                                fig.update_yaxes(range=[0, df_viz[viz['y_axis']].max() * 1.1])
                            elif viz['chart_type'] == "Histogram":
                                fig = px.histogram(df_viz, x=viz['x_axis'], nbins=20, title=f"Histogram of {viz['x_axis']}")
                            elif viz['chart_type'] == "Heatmap":
                                corr = df_viz[numeric_columns].corr()
                                fig, ax = plt.subplots(figsize=(8, 5))
                                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                                st.pyplot(fig)
                            if viz['chart_type'] != "Heatmap":
                                st.plotly_chart(fig, use_container_width=True, key=f"dashboard_viz_{viz['id']}")
                        except Exception as e:
                            st.error(f"Error generating visualization {i+1}: {str(e)}")

                    if st.button("Remove Visualization", key=f"remove_dashboard_{viz['id']}"):
                        visualizations.pop(i)
                        if f"filtered_data_{viz['id']}" in st.session_state:
                            del st.session_state[f"filtered_data_{viz['id']}"]
                        for key in list(st.session_state.active_filters['dashboard'].keys()):
                            if key.startswith(f"viz_{i+1}_"):
                                del st.session_state.active_filters['dashboard'][key]
                        st.rerun()

    with tab5:
        st.subheader("🔍 Insights & Filtered Data")
        st.write("View the dataset after dashboard filters and explore AI-generated insights.")

        st.write("### Filtered Data View")
        st.write("View the dataset after dashboard filters (Global or Per-Visualization). Apply additional filters here without affecting other tabs.")
        
        filtered_df = st.session_state.df_dashboard
        with st.expander("Apply Additional Filters", expanded=False):
            view_categorical_filters = {}
            for col in categorical_columns:
                selected_vals = st.multiselect(f"Filter by {col}", options=filtered_df[col].unique(), key=f"view_cat_{col}")
                view_categorical_filters[col] = selected_vals
                if selected_vals:
                    st.session_state.active_filters['filtered_view'][f"{col}_categorical"] = selected_vals

            view_numeric_filters = {}
            for col in numeric_columns:
                min_val, max_val = float(filtered_df[col].min()), float(filtered_df[col].max())
                if min_val != max_val:
                    selected_range = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val), key=f"view_num_{col}")
                    view_numeric_filters[col] = selected_range
                    if selected_range != (min_val, max_val):
                        st.session_state.active_filters['filtered_view'][f"{col}_numeric"] = selected_range
                else:
                    st.write(f"{col}: Constant value {min_val}")
                    view_numeric_filters[col] = (min_val, max_val)

            view_date_filter = {}
            if date_columns:
                view_date_filter['date_col'] = st.selectbox("Select Date Column", date_columns, key="view_date_col")
                view_date_filter['date_format'] = st.text_input("Specify Date Format (e.g., %Y-%m-%d)", value=None, key="view_date_format")
                view_date_filter['use_date_filter'] = st.checkbox("Filter by Date", key="view_date_filter")
                if view_date_filter['use_date_filter']:
                    view_date_filter['start_date'] = st.date_input("Start Date", filtered_df[view_date_filter['date_col']].min().date(), key="view_start_date")
                    view_date_filter['end_date'] = st.date_input("End Date", filtered_df[view_date_filter['date_col']].max().date(), key="view_end_date")
                    st.session_state.active_filters['filtered_view']['date_filter'] = f"{view_date_filter['date_col']} from {view_date_filter['start_date']} to {view_date_filter['end_date']}"

            view_query = st.text_area("Enter custom query (e.g., Value > 100)", height=100, key="view_query")
            if view_query:
                st.session_state.active_filters['filtered_view']['custom_query'] = view_query

            if st.button("Preview Additional Filters", key="preview_view_filters"):
                df_preview = apply_filters(st.session_state.df_dashboard.copy(), view_categorical_filters, view_numeric_filters, view_date_filter, view_query)
                if df_preview.empty:
                    st.warning("The applied additional filters result in an empty dataset. Please adjust your filters or query.")
                else:
                    st.dataframe(df_preview.head(100), use_container_width=True)

            if st.button("Apply Additional Filters", key="apply_view_filters"):
                filtered_df = apply_filters(st.session_state.df_dashboard.copy(), view_categorical_filters, view_numeric_filters, view_date_filter, view_query)
                st.session_state.filtered_df = filtered_df
                if filtered_df.empty:
                    st.warning("Additional filters result in an empty dataset. Please adjust your filters or query.")
                else:
                    st.success("Additional filters applied successfully!")

            if st.button("Reset Additional Filters", key="reset_view_filters"):
                filtered_df = st.session_state.df_dashboard.copy()
                st.session_state.filtered_df = filtered_df
                st.session_state.active_filters['filtered_view'] = {}
                st.success("Additional filters reset.")

        if st.session_state.selected_filter_mode == "Global Dashboard Filters":
            if filtered_df.empty:
                st.warning("The filtered dataset is empty due to global dashboard filters.")
            else:
                st.write("### Filtered Data (Global Dashboard Filters)")
                st.dataframe(filtered_df.head(100), use_container_width=True)
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name="global_filtered_data.csv",
                    mime="text/csv",
                    key="download_global_filtered"
                )
        else:
            visualizations = st.session_state.default_visualizations if st.session_state.dashboard_mode == "Default Dashboard" else st.session_state.custom_visualizations
            if not visualizations:
                st.info("No visualizations available. Add visualizations in the Filters & Visualization tab to view filtered data.")
            else:
                viz_options = [f"Visualization {i+1}: {viz['chart_type']}" for i, viz in enumerate(visualizations)]
                selected_viz = st.selectbox("Select Visualization to View Filtered Data", viz_options, key="select_viz")
                viz_index = viz_options.index(selected_viz)
                viz = visualizations[viz_index]
                filtered_df = st.session_state.get(f"filtered_data_{viz['id']}", df.copy())
                filtered_df = apply_filters(filtered_df, view_categorical_filters, view_numeric_filters, view_date_filter, view_query)
                if filtered_df.empty:
                    st.warning(f"The filtered dataset for {selected_viz} is empty due to the applied filters or query.")
                else:
                    st.write(f"### Filtered Data for {selected_viz}")
                    st.dataframe(filtered_df.head(100), use_container_width=True)
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Filtered Data as CSV",
                        data=csv,
                        file_name=f"filtered_data_viz_{viz_index+1}.csv",
                        mime="text/csv",
                        key=f"download_viz_{viz_index}"
                    )

        st.write("### AI Insights: Data Summary")
        with st.expander("Show Summary", expanded=False):
            filtered_df = st.session_state.filtered_df
            st.write(f"The dataset contains {filtered_df.shape[0]} rows and {filtered_df.shape[1]} columns.")
            st.write("#### Numeric Columns")
            for col in numeric_columns:
                mean = filtered_df[col].mean()
                min_val = filtered_df[col].min()
                max_val = filtered_df[col].max()
                skewness = filtered_df[col].skew()
                st.write(f"{col}: Mean = {mean:.2f}, Min = {min_val:.2f}, Max = {max_val:.2f}, Skewness = {skewness:.2f}")
                if abs(skewness) > 1:
                    st.write(f"⚠️ {col} is highly skewed. Consider transformation (e.g., log).")
            st.write("#### Categorical Columns")
            for col in categorical_columns:
                st.write(f"{col}: {filtered_df[col].nunique()} unique values")
            if len(numeric_columns) >= 2:
                st.write("#### Key Correlations")
                corr = filtered_df[numeric_columns].corr()
                high_corr = corr[(corr > 0.7) | (corr < -0.7)]
                for col1 in high_corr.columns:
                    for col2 in high_corr.index:
                        if col1 < col2 and not pd.isna(high_corr.loc[col2, col1]) and abs(high_corr.loc[col2, col1]) > 0.7:
                            st.write(f"Strong correlation between {col1} and {col2}: {high_corr.loc[col2, col1]:.2f}")

else:
    st.warning("Please upload a CSV file to get started.")