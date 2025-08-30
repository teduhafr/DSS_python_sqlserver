import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
## menggunakan python 3.11, di pc menggunakan conda dss. pakai cmd. kalau power shell error.
# --- Page Configuration ---
st.set_page_config(page_title="Dynamic DB Pivot with User Input", layout="wide")

# --- Initialize Session State ---
if 'connected' not in st.session_state:
    st.session_state.connected = False
    st.session_state.engine = None
    st.session_state.table = ""
    st.session_state.available_columns = []
    st.session_state.tables = []
    st.session_state.result_df = None
    st.session_state.preview_df = None
    st.session_state.pivot_params = {}
    st.session_state.drill_down_df = None
    st.session_state.drill_down_info = None
    st.session_state.searchable_columns = []
    st.session_state.numeric_columns = []

# --- Database Connection Function ---
def get_sql_server_engine(server, database, username, password):
    try:
        driver = "ODBC Driver 17 for SQL Server"
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
        engine = create_engine(conn_str, connect_args={"timeout": 10})
        connection = engine.connect()
        connection.close()
        return engine
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {e}")
        return None

# --- Drill-Down Query ---
def fetch_drill_down_data(table_name, pivot_params, clicked_row_data, clicked_col_name):
    drill_down_col_value = clicked_col_name
    pivot_col = pivot_params.get('pivot_col')
    if not pivot_col:
        st.warning("Pivot column information not found for drill-down.")
        return None, None

    if clicked_col_name == '(Empty)':
        drill_down_col_value = ''

    filter_desc = {
        **{col: clicked_row_data.get(col) for col in pivot_params['rows']},
        pivot_col: drill_down_col_value
    }

    is_total_row = pivot_params.get('show_col_totals') and pivot_params['rows'] and clicked_row_data.get(pivot_params['rows'][0]) == 'Total'
    is_total_col = pivot_params.get('show_row_totals') and clicked_col_name == 'Total'

    try:
        where_clauses = []
        params = {}

        if not is_total_row:
            for col in pivot_params['rows']:
                if col in clicked_row_data and clicked_row_data[col] is not None:
                    param_name = f"param_{col.replace(' ', '_')}"
                    where_clauses.append(f'"{col}" = :{param_name}')
                    params[param_name] = clicked_row_data[col]

        if not is_total_col:
            param_name = f"param_{pivot_col.replace(' ', '_')}"
            params[param_name] = clicked_col_name # Use the name from the pivot column, e.g., '(Empty)'
            where_clauses.append(f"COALESCE(NULLIF(LTRIM(RTRIM(\"{pivot_col}\")), ''), '(Empty)') = :{param_name}")

        where_str = f' WHERE {" AND ".join(where_clauses)}' if where_clauses else ''
        query_str = f'SELECT * FROM "{table_name}"{where_str}'
        query = text(query_str)

        with st.session_state.engine.connect() as connection:
            drill_df = pd.read_sql(query, connection, params=params)
        
        return drill_df, filter_desc
    except Exception as e:
        st.error(f"Could not fetch drill-down data: {e}")
        return None, None

# --- Sidebar: Database Connection ---
with st.sidebar:
    st.title("🔗 Database Connection")
    server = st.text_input("Server/Host", placeholder="your_server.database.windows.net")
    database = st.text_input("Database", placeholder="your_database_name")
    username = st.text_input("Username", placeholder="your_username")
    password = st.text_input("Password", type="password")

    if st.button("Connect"):
        if not all([server, database, username, password]):
            st.warning("Please fill in all connection details.")
        else:
            with st.spinner("Connecting to database..."):
                engine = get_sql_server_engine(server, database, username, password)
                if engine:
                    st.session_state.engine = engine
                    st.session_state.connected = True
                    inspector = inspect(engine)
                    st.session_state.tables = inspector.get_table_names()
                    st.success("Connection successful!")
                else:
                    st.session_state.connected = False
                    st.session_state.engine = None

# --- Main UI ---
if st.session_state.connected:
    with st.sidebar:
        st.markdown("---")
        prev_table = st.session_state.table
        st.session_state.table = st.selectbox("Select Table", options=st.session_state.tables)

        if prev_table != st.session_state.table:
            st.session_state.result_df = None
            st.session_state.preview_df = None
            st.session_state.drill_down_df = None
            st.session_state.drill_down_info = None

        if st.session_state.table:
            try:
                with st.session_state.engine.connect() as connection:
                    sample_df = pd.read_sql(f'SELECT TOP 1 * FROM "{st.session_state.table}"', connection)
                    st.session_state.available_columns = sample_df.columns.tolist()
                    inspector = inspect(st.session_state.engine)
                    columns_info = inspector.get_columns(st.session_state.table)
                    st.session_state.searchable_columns = [
                        c['name'] for c in columns_info
                        if 'char' in str(c['type']).lower() or 'text' in str(c['type']).lower()
                    ]
                    st.session_state.numeric_columns = [
                        c['name'] for c in columns_info
                        if any(t in str(c['type']).lower() for t in ['int', 'float', 'decimal', 'numeric', 'money'])
                    ]
            except Exception as e:
                st.error(f"Could not fetch columns for table '{st.session_state.table}': {e}")
                st.session_state.connected = False

        # Pivot controls
        st.header("⚙️ Pivot Controls")
        if st.session_state.available_columns:
            cols_options = st.session_state.available_columns[:]
            rows = st.multiselect("Rows", options=cols_options, default=cols_options[0] if cols_options else None)
            pivot_col = st.selectbox("Columns", options=cols_options, index=1 if len(cols_options) > 1 else 0)
            value = st.selectbox("Value", options=cols_options, index=2 if len(cols_options) > 2 else 0)
            agg_func = st.selectbox("Aggregation", options=['SUM', 'AVG', 'COUNT', 'MAX', 'MIN'])

            st.markdown("---")
            st.write("📊 **Totals**")
            st.checkbox("Show Row Totals (horizontal)", key="show_row_totals")
            st.checkbox("Show Column Totals (vertical)", key="show_col_totals")

    tab1, tab2 = st.tabs(["⚡ Pivot Generator", "🔍 Data Explorer"])

    with tab1:
        st.title("⚡ Dynamic Pivot Table Generator")

        # === Generate Pivot ===
        if st.button("🚀 Generate Pivot Table"):
            if not all([rows, pivot_col, value, st.session_state.table]):
                st.warning("Please select a table, at least one row, a column, and a value.")
            else:
                # --- Aggregation Validation ---
                is_numeric_agg = agg_func in ['SUM', 'AVG']
                value_col_is_numeric = value in st.session_state.get('numeric_columns', [])

                if is_numeric_agg and not value_col_is_numeric:
                    st.error(f"Aggregation '{agg_func}' requires a numeric 'Value' column, but '{value}' is not a numeric column. Please choose a numeric column for the 'Value' field or select a different aggregation like 'COUNT'.")
                else:
                    st.session_state.drill_down_df = None
                    st.session_state.drill_down_info = None
                    try:
                        st.session_state.pivot_params = {
                            "rows": rows,
                            "pivot_col": pivot_col,
                            "value": value,
                            "agg_func": agg_func,
                            "show_row_totals": st.session_state.get("show_row_totals", False),
                            "show_col_totals": st.session_state.get("show_col_totals", False)
                        }
                        st.session_state.result_df = None

                        with st.spinner("Generating SQL and executing pivot query..."):
                            with st.session_state.engine.connect() as connection:
                                # 1. Get distinct values for the pivot column
                                distinct_query = text(f"""
                                    SELECT DISTINCT COALESCE(NULLIF(LTRIM(RTRIM("{pivot_col}")), ''), '(Empty)') AS "{pivot_col}"
                                    FROM "{st.session_state.table}"
                                    ORDER BY "{pivot_col}";
                                """)
                                result_proxy = connection.execute(distinct_query)
                                distinct_values_df = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys())
                                pivot_values = [str(val) for val in distinct_values_df[pivot_col].tolist() if val]

                                if not pivot_values:
                                    st.warning(f"No distinct values found for column '{pivot_col}'. Pivot cannot be generated.")
                                    st.stop()

                                # 2. Build the dynamic SQL query with CASE statements
                                show_row_totals = st.session_state.pivot_params['show_row_totals']
                                show_col_totals = st.session_state.pivot_params['show_col_totals']
                                row_cols_str = ", ".join([f'"{r}"' for r in rows])

                                agg_expressions = []
                                for v in pivot_values:
                                    safe_alias = v.strip().replace('"', '""')
                                    alias = f'"{safe_alias}"'
                                    condition_value = '' if v == '(Empty)' else v
                                    condition_value = condition_value.replace("'", "''") # Basic SQL injection guard for values
                                    expression = f'{agg_func}(CASE WHEN "{pivot_col}" = \'{condition_value}\' THEN "{value}" END) AS {alias}'
                                    agg_expressions.append(expression)

                                if show_row_totals:
                                    agg_expressions.append(f'{agg_func}("{value}") AS "Total"')

                                agg_expressions_str = ",\n       ".join(agg_expressions)
                                select_cols = row_cols_str
                                grouping_logic = f"GROUP BY {row_cols_str}" if rows else ""

                                if show_col_totals and rows:
                                    grouping_logic = f"GROUP BY GROUPING SETS (({row_cols_str}), ())"
                                    coalesce_expressions = []
                                    coalesce_expressions.append(f"COALESCE(CAST(\"{rows[0]}\" AS VARCHAR(MAX)), 'Total') AS \"{rows[0]}\"")
                                    for r in rows[1:]:
                                        coalesce_expressions.append(f"COALESCE(CAST(\"{r}\" AS VARCHAR(MAX)), '') AS \"{r}\"")
                                    select_cols = ", ".join(coalesce_expressions)

                                dynamic_pivot_query_str = f"""
                                    SELECT
                                        {select_cols}{',' if select_cols and agg_expressions_str else ''}
                                        {agg_expressions_str}
                                    FROM "{st.session_state.table}"
                                    {grouping_logic};
                                """
                                st.subheader("Generated T-SQL Query")
                                st.code(dynamic_pivot_query_str, language="sql")

                                dynamic_pivot_query = text(dynamic_pivot_query_str)
                                result_proxy_final = connection.execute(dynamic_pivot_query)
                                st.session_state.result_df = pd.DataFrame(result_proxy_final.fetchall(), columns=result_proxy_final.keys())
                                st.session_state.result_df.columns.name = st.session_state.pivot_params['pivot_col']

                        st.success(f"Pivot generated successfully! Found {st.session_state.result_df.shape[0]} rows.")
                    except Exception as e:
                        st.error(f"An error occurred during pivot generation: {e}")
                        st.session_state.result_df = None

        # --- Display Pivot Result ---
        if st.session_state.result_df is not None and not st.session_state.result_df.empty:
            st.markdown("---")
            
            # Build dynamic title for the pivot result
            params = st.session_state.pivot_params
            agg_func = params.get('agg_func', 'Aggregation')
            value_col = params.get('value', 'Value')
            row_cols_str = ", ".join(params.get('rows', []))
            pivot_col = params.get('pivot_col', 'Column')
            
            title = f"📈 {agg_func} of `{value_col}` by `{row_cols_str}` across `{pivot_col}`"
            st.subheader(title)

            df_to_filter = st.session_state.result_df.copy()

            # --- Filter for Pivot Result (Collapsible) ---
            with st.expander("🔎 Filter Pivot Results", expanded=False):
                # --- Column Filter ---
                row_cols = st.session_state.pivot_params.get('rows', [])
                value_cols = [c for c in df_to_filter.columns if c not in row_cols]

                selected_display_cols = st.multiselect(
                    "Show/Hide Pivot Columns",
                    options=value_cols,
                    default=value_cols
                )
                # Reconstruct the dataframe with the selected columns, keeping row columns at the front
                df_to_filter = df_to_filter[row_cols + selected_display_cols]

                st.markdown("---")
                st.write("Apply up to 10 cascading row filters. Filters are combined with AND.")
                num_pivot_filters = 10
                # The filterable "row columns" are the ones selected by the user
                available_cols = st.session_state.pivot_params.get('rows', [])

                for i in range(num_pivot_filters):
                    # Use columns to layout the filter widgets
                    cols = st.columns([1, 2])
                    with cols[0]:
                        filter_column = st.selectbox(
                            f"Filter column #{i+1}",
                            options=['-- Select a row column --'] + available_cols,
                            key=f"pivot_filter_col_{i}"
                        )

                    if filter_column != '-- Select a row column --':
                        try:
                            is_numeric = pd.api.types.is_numeric_dtype(df_to_filter[filter_column])
                            with cols[1]:
                                if is_numeric:
                                    # Drop NA to prevent errors with min/max on columns with mixed types or NaNs
                                    col_series = df_to_filter[filter_column].dropna()
                                    if not col_series.empty:
                                        min_val = float(col_series.min())
                                        max_val = float(col_series.max())
                                        if min_val == max_val:
                                            st.info(f"Column '{filter_column}' has only one numeric value: {min_val}")
                                        else:
                                            range_val = st.slider(
                                                f"Select range for '{filter_column}'",
                                                min_value=min_val,
                                                max_value=max_val,
                                                value=(min_val, max_val),
                                                key=f"pivot_filter_slider_{i}"
                                            )
                                            start_range, end_range = range_val
                                            # Filter the dataframe based on the slider's range.
                                            df_to_filter = df_to_filter[df_to_filter[filter_column].between(start_range, end_range)]
                                    else:
                                        st.info(f"Column '{filter_column}' contains no numeric data to filter.")
                                else: # Categorical column
                                    # Get unique values from the *currently filtered* dataframe to make filters dependent
                                    unique_values = sorted(df_to_filter[filter_column].dropna().astype(str).unique())
                                    selected_values = st.multiselect(f"Select values for '{filter_column}'", options=unique_values, key=f"pivot_filter_values_{i}")
                                    if selected_values:
                                        df_to_filter = df_to_filter[df_to_filter[filter_column].astype(str).isin(selected_values)]
                        except Exception as e:
                            st.warning(f"Could not apply filter on '{filter_column}': {e}")

            # --- Interactive AG-Grid Display ---
            gb = GridOptionsBuilder.from_dataframe(df_to_filter)
            gb.configure_default_column(
                resizable=True,
                filterable=True,
                sortable=True,
                editable=False,
            )

            # Pin the row columns to the left for better navigation
            for col in row_cols:
                gb.configure_column(col, pinned='left')

            # Enable single row selection
            gb.configure_selection(selection_mode='single', use_checkbox=False)

            gridOptions = gb.build()

            grid_response = AgGrid(
                df_to_filter,
                gridOptions=gridOptions,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=False,
                height=400,
                width='100%',
                key='pivot_grid'
            )

            # --- Interactive Drill-Down on Row Selection ---
            selected_rows = grid_response.get("selected_rows")

            if selected_rows is not None and not selected_rows.empty:
                selected_row_data = selected_rows.iloc[0].to_dict()

                # Create a descriptive label for the selected row
                row_cols = st.session_state.pivot_params.get('rows', [])
                row_label_parts = [f"{col}: {selected_row_data.get(col, 'N/A')}" for col in row_cols]
                row_label = " | ".join(row_label_parts)

                st.markdown("---")
                st.write(f"#### Select a column to drill down into for row: **{row_label}**")

                # Get numeric columns that are not part of the row definition
                numeric_cols = [c for c in df_to_filter.columns if pd.api.types.is_numeric_dtype(df_to_filter[c]) and c not in row_cols]

                if not numeric_cols:
                    st.warning("No numeric data columns available in the current view to drill down into.")
                else:
                    with st.form(key='drill_down_form'):
                        col_name = st.selectbox("Drill-down column:", numeric_cols)
                        submitted = st.form_submit_button("Drill Down")

                        if submitted:
                            is_total_col = col_name == 'Total'
                            is_total_row = selected_row_data.get(row_cols[0] if row_cols else None) == 'Total'

                            if not is_total_col and not is_total_row:
                                with st.spinner(f"Fetching details for '{col_name}'..."):
                                    drill_df, filter_desc = fetch_drill_down_data(
                                        st.session_state.table,
                                        st.session_state.pivot_params,
                                        selected_row_data,
                                        col_name
                                    )
                                    st.session_state.drill_down_df = drill_df
                                    st.session_state.drill_down_info = filter_desc
                                    st.rerun()
                            else:
                                st.toast("Drill-down is not available for 'Total' rows or columns.")

            # --- Drill-Down Display ---
            if st.session_state.get('drill_down_df') is not None:
                st.markdown("---")
                st.subheader("🔬 Drill-Down Details")

                if st.session_state.drill_down_info:
                    st.write("Showing raw data for:")
                    st.json(st.session_state.drill_down_info)

                if not st.session_state.drill_down_df.empty:
                    st.dataframe(st.session_state.drill_down_df, use_container_width=True)
                    st.info(f"Displaying {len(st.session_state.drill_down_df)} raw data rows.")
                else:
                    st.warning("No underlying data found for the selected cell.")

                if st.button("Hide Details"):
                    st.session_state.drill_down_df = None
                    st.session_state.drill_down_info = None
                    st.rerun()

            # --- Chart Generator (Collapsible) ---
            with st.expander("📊 Chart Generator", expanded=False):
                chart_df = df_to_filter.copy() # The dataframe is already flat

                row_cols = st.session_state.pivot_params.get('rows', [])

                # Exclude 'Total' row from charting if it exists, as it can skew visualizations
                if st.session_state.pivot_params.get('show_col_totals') and row_cols and not chart_df.empty and chart_df[row_cols[0]].iloc[-1] == 'Total':
                    chart_df = chart_df.iloc[:-1]

                if not chart_df.empty and row_cols:
                    chart_type = st.selectbox(
                        "Select Chart Type",
                        ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"]
                    )

                    if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                        x_axis = st.selectbox("Select X-axis", options=row_cols, key="chart_x_axis")

                        all_value_cols = [c for c in chart_df.columns if c not in row_cols]
                        y_axis = st.multiselect("Select Y-axis/axes", options=all_value_cols, default=all_value_cols[0] if all_value_cols else None, key="chart_y_axis")

                        if x_axis and y_axis:
                            try:
                                title = f"{chart_type}: {', '.join(y_axis)} by {x_axis}"
                                if chart_type == "Bar Chart":
                                    fig = px.bar(chart_df, x=x_axis, y=y_axis, title=title, barmode='group')
                                elif chart_type == "Line Chart":
                                    fig = px.line(chart_df, x=x_axis, y=y_axis, title=title)
                                elif chart_type == "Scatter Plot":
                                    fig = px.scatter(chart_df, x=x_axis, y=y_axis, title=title)

                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not generate chart: {e}")

                    elif chart_type == "Pie Chart":
                        label_col = st.selectbox("Select Labels", options=row_cols, key="pie_label")
                        value_cols = [c for c in chart_df.columns if c not in row_cols and pd.api.types.is_numeric_dtype(chart_df[c])]
                        if value_cols:
                            value_col = st.selectbox("Select Values", options=value_cols, key="pie_value")

                            if label_col and value_col:
                                try:
                                    fig = px.pie(chart_df, names=label_col, values=value_col, title=f"Pie Chart: {value_col} by {label_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not generate chart: {e}")
                        else:
                            st.warning("No numeric value columns available for a Pie Chart.")
                elif not row_cols:
                     st.info("Select at least one 'Row' in pivot controls to generate a chart.")
                else:
                    st.info("No data available for charting (the 'Total' row is excluded).")

    with tab2:
        st.title("🔍 Data Explorer")

        filterable_columns = st.session_state.get('searchable_columns', []) + st.session_state.get('numeric_columns', [])

        if st.session_state.table and filterable_columns:
            st.subheader(f"Explore and Filter '{st.session_state.table}'")

            st.write("Filter data using up to 5 conditions. Filters are combined with AND.")

            filters = []
            num_filters = 5
            for i in range(num_filters):
                cols = st.columns([2, 3])
                with cols[0]:
                    col_name = st.selectbox(
                        f"Column #{i+1}",
                        options=["-- None --"] + sorted(filterable_columns),
                        key=f"de_filter_col_{i}"
                    )

                if col_name != "-- None --":
                    is_numeric = col_name in st.session_state.get('numeric_columns', [])
                    with cols[1]:
                        if is_numeric:
                            val_cols = st.columns(2)
                            val1 = val_cols[0].number_input("From", key=f"de_val1_{i}", value=None)
                            val2 = val_cols[1].number_input("To", key=f"de_val2_{i}", value=None)

                            if val1 is not None and val2 is not None:
                                filters.append({"column": col_name, "type": "numeric_range", "value": (val1, val2)})
                            elif val1 is not None:
                                filters.append({"column": col_name, "type": "numeric_gte", "value": val1})
                            elif val2 is not None:
                                filters.append({"column": col_name, "type": "numeric_lte", "value": val2})
                        else:  # Text column
                            val = st.text_input("Contains value", key=f"de_filter_val_{i}", placeholder="Type to search...")
                            if val:
                                filters.append({"column": col_name, "type": "text_contains", "value": val})

            if st.button("Show Data / Apply Filter", key="apply_filter"):
                with st.spinner("Fetching data..."):
                    try:
                        with st.session_state.engine.connect() as connection:
                            query_str = f'SELECT * FROM "{st.session_state.table}"'
                            params = {}
                            if filters:
                                where_clauses = []
                                for i, f in enumerate(filters):
                                    col = f['column']
                                    if f['type'] == 'text_contains':
                                        param_name = f"param_{i}"
                                        where_clauses.append(f'"{col}" LIKE :{param_name}')
                                        params[param_name] = f"%{f['value']}%"
                                    elif f['type'] == 'numeric_gte':
                                        param_name = f"param_{i}"
                                        where_clauses.append(f'"{col}" >= :{param_name}')
                                        params[param_name] = f['value']
                                    elif f['type'] == 'numeric_lte':
                                        param_name = f"param_{i}"
                                        where_clauses.append(f'"{col}" <= :{param_name}')
                                        params[param_name] = f['value']
                                    elif f['type'] == 'numeric_range':
                                        param_name1, param_name2 = f"param_{i}_a", f"param_{i}_b"
                                        where_clauses.append(f'"{col}" BETWEEN :{param_name1} AND :{param_name2}')
                                        params[param_name1], params[param_name2] = f['value']

                                query_str += f' WHERE {" AND ".join(where_clauses)}'

                            query = text(query_str)
                            st.session_state.preview_df = pd.read_sql(query, connection, params=params)
                    except Exception as e:
                        st.error(f"Failed to fetch data: {e}")
                        st.session_state.preview_df = None

            if st.session_state.preview_df is not None:
                st.markdown("---")
                st.dataframe(st.session_state.preview_df, use_container_width=True)
                st.info(f"Showing {len(st.session_state.preview_df)} rows.")

        else:
            if not st.session_state.table:
                st.info("Select a table from the sidebar to explore its data.")
            else:
                st.warning(f"The selected table '{st.session_state.table}' has no text or numeric columns available for filtering.")

else:
    st.info("Please enter your database credentials and click 'Connect' in the sidebar to begin.")