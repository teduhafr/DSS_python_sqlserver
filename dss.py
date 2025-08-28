import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect

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

# --- Database Connection Function ---
def get_sql_server_engine(server, database, username, password):
    """Creates a SQLAlchemy engine from user inputs and tests the connection."""
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

def fetch_drill_down_data(table_name, pivot_params, clicked_row_data, clicked_col_name):
    """Fetch raw data for a pivot cell based on selected filters."""
    # Reverse the mapping from the display column name to the actual data value.
    # The pivot query aliases empty strings as '(Empty)' because empty strings are invalid column names.
    drill_down_col_value = clicked_col_name
    if clicked_col_name == '(Empty)':
        drill_down_col_value = ''

    filter_desc = {
        **{col: clicked_row_data.get(col) for col in pivot_params['rows']},
        pivot_params['pivot_col']: drill_down_col_value
    }

    is_total_row = pivot_params.get('show_col_totals') and pivot_params['rows'] and clicked_row_data.get(pivot_params['rows'][0]) == 'Total'
    is_total_col = pivot_params.get('show_row_totals') and clicked_col_name == 'Total'

    try:
        where_clauses = []
        params = {}

        if not is_total_row:
            for col in pivot_params['rows']:
                # Ensure the row data for this column exists before adding filter
                if col in clicked_row_data and clicked_row_data[col] is not None:
                    param_name = f"param_{col.replace(' ', '_')}"
                    where_clauses.append(f'"{col}" = :{param_name}')
                    params[param_name] = clicked_row_data[col]

        if not is_total_col:
            pivot_col = pivot_params['pivot_col']
            param_name = f"param_{pivot_col.replace(' ', '_')}"
            where_clauses.append(f'"{pivot_col}" = :{param_name}')
            params[param_name] = drill_down_col_value

        where_str = f' WHERE {" AND ".join(where_clauses)}' if where_clauses else ''
        query_str = f'SELECT * FROM "{table_name}"{where_str}'
        query = text(query_str)

        with st.session_state.engine.connect() as connection:
            drill_df = pd.read_sql(query, connection, params=params)
        
        return drill_df, filter_desc
    except Exception as e:
        st.error(f"Could not fetch drill-down data: {e}")
        return None, None

# --- UI: Connection Details ---
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

# --- Main App ---
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
            except Exception as e:
                st.error(f"Could not fetch columns for table '{st.session_state.table}': {e}")
                st.session_state.connected = False

        # --- Pivot Controls ---
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

    st.title("⚡ Dynamic Pivot Table Generator")

    # === Original Data Preview ===
    if st.session_state.table:
        with st.expander("📄 View Original Data Preview"):
            rows_to_preview = st.number_input("Number of rows to preview", min_value=10, max_value=100000, value=100, step=10)

            st.markdown("##### 🔎 Search Before Loading")
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                column_to_search = st.selectbox("Search in column", options=["All Text Columns"] + st.session_state.get('searchable_columns', []), key="preview_search_col")
            with col2:
                search_term = st.text_input("Enter search term (optional)", key="preview_search_term", placeholder="Case-insensitive search...")

            if st.button("Load Preview Data"):
                st.session_state.preview_df = None
                try:
                    spinner_msg = f"Loading top {rows_to_preview} rows from '{st.session_state.table}'..."
                    if search_term:
                        spinner_msg = f"Searching for '{search_term}' and loading top {rows_to_preview} rows..."

                    with st.spinner(spinner_msg):
                        with st.session_state.engine.connect() as connection:
                            query_string = f'SELECT TOP {rows_to_preview} * FROM "{st.session_state.table}"'
                            params = {}

                            if search_term:
                                params['search_term'] = f"%{search_term}%"
                                searchable_cols = st.session_state.get('searchable_columns', [])

                                if column_to_search == "All Text Columns" and searchable_cols:
                                    where_clauses = [f'CAST("{col}" AS VARCHAR(MAX)) LIKE :search_term' for col in searchable_cols]
                                    query_string += " WHERE " + " OR ".join(where_clauses)
                                elif column_to_search != "All Text Columns":
                                    query_string += f' WHERE CAST("{column_to_search}" AS VARCHAR(MAX)) LIKE :search_term'

                            query = text(query_string)
                            st.session_state.preview_df = pd.read_sql(query, connection, params=params)
                except Exception as e:
                    st.error(f"Failed to load original data: {e}")
                    st.session_state.preview_df = None

            if st.session_state.preview_df is not None and not st.session_state.preview_df.empty:
                st.success(f"Found and loaded {len(st.session_state.preview_df)} rows.")
                df_for_display = st.session_state.preview_df

                total_rows = len(df_for_display)
                col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
                with col1:
                    page_size = st.number_input("Rows per page", min_value=10, max_value=200, value=50, step=10, key="preview_page_size")
                total_pages = max(1, (total_rows // page_size) + (1 if total_rows % page_size > 0 else 0))
                with col2:
                    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="preview_page_number")

                start_idx = (page_number - 1) * page_size
                end_idx = start_idx + page_size
                df_to_display = df_for_display.iloc[start_idx:end_idx]

                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    info_text = f"Showing rows {start_idx+1}-{min(end_idx, total_rows)} of {total_rows} found"
                    st.info(info_text)

                st.dataframe(df_to_display, use_container_width=True)

    st.markdown("---")

    # --- Generate Pivot Table ---
    if st.button("🚀 Generate Pivot Table"):
        if not all([rows, pivot_col, value, st.session_state.table]):
            st.warning("Please select a table, rows, a column, and a value.")
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

                with st.session_state.engine.connect() as connection:
                    # ✅ Get distinct values for pivot column
                    distinct_query = text(f"""
                        SELECT DISTINCT COALESCE(NULLIF(LTRIM(RTRIM(CAST("{pivot_col}" AS VARCHAR(MAX)))), ''), '(Empty)') AS "{pivot_col}"
                        FROM "{st.session_state.table}"
                        ORDER BY "{pivot_col}";
                    """)
                    result_proxy = connection.execute(distinct_query)
                    distinct_values_df = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys())
                    pivot_values = [str(val) for val in distinct_values_df[pivot_col].tolist() if val]

                    if not pivot_values:
                        st.warning(f"No distinct values found for column '{pivot_col}'. Pivot cannot be generated.")
                    else:
                        show_row_totals = st.session_state.pivot_params['show_row_totals']
                        show_col_totals = st.session_state.pivot_params['show_col_totals']
                        row_cols_str = ", ".join([f'"{r}"' for r in rows])

                        # ✅ Build CASE expressions for conditional aggregation
                        agg_expressions = []
                        for v in pivot_values:
                            # Skip empty values that would create invalid column aliases
                            if not v or v.strip() == '':
                                continue
                                
                            safe_alias = v.strip()
                            # Replace any quotes in alias and ensure it's not empty
                            safe_alias = safe_alias.replace('"', '""')
                            
                            # Skip if alias would still be empty after cleaning
                            if not safe_alias:
                                continue
                                
                            alias = f'"{safe_alias}"'

                            # Use original value for CASE condition (replace '(Empty)' with actual empty string)
                            condition_value = '' if v == '(Empty)' else v
                            condition_value = condition_value.replace("'", "''")  # escape single quotes

                            # Handle empty string condition properly
                            if condition_value == '':
                                condition = f'COALESCE(NULLIF(LTRIM(RTRIM(CAST("{pivot_col}" AS VARCHAR(MAX)))), \'\'), \'(Empty)\') = \'(Empty)\''
                            else:
                                condition = f'COALESCE(NULLIF(LTRIM(RTRIM(CAST("{pivot_col}" AS VARCHAR(MAX)))), \'\'), \'(Empty)\') = \'{condition_value}\''

                            expression = f'{agg_func}(CASE WHEN {condition} THEN "{value}" END) AS {alias}'
                            agg_expressions.append(expression)

                        # ✅ Add row total column if requested
                        if show_row_totals:
                            agg_expressions.append(f'{agg_func}("{value}") AS "Total"')

                        if not agg_expressions:
                            st.warning(f"No valid pivot values found for column '{pivot_col}'. All values appear to be empty or invalid.")
                        else:
                            agg_expressions_str = ",\n       ".join(agg_expressions)

                            select_cols = row_cols_str
                            grouping_logic = f"GROUP BY {row_cols_str}" if rows else ""

                            # ✅ Handle column totals (grand total row)
                            if show_col_totals and rows:
                                grouping_logic = f"GROUP BY GROUPING SETS (({row_cols_str}), ())"
                                coalesce_expressions = []
                                coalesce_expressions.append(f"COALESCE(CAST(\"{rows[0]}\" AS VARCHAR(MAX)), 'Total') AS \"{rows[0]}\"")
                                for r in rows[1:]:
                                    coalesce_expressions.append(f"COALESCE(CAST(\"{r}\" AS VARCHAR(MAX)), '') AS \"{r}\"")
                                select_cols = ", ".join(coalesce_expressions)

                            # ✅ Build the final SQL
                            dynamic_pivot_query_str = f"""
                                SELECT
                                    {select_cols}{',' if select_cols and agg_expressions_str else ''}
                                    {agg_expressions_str}
                                FROM "{st.session_state.table}"
                                {grouping_logic};
                            """

                            # Show generated SQL in UI
                            st.subheader("Generated T-SQL Query")
                            st.code(dynamic_pivot_query_str, language="sql")

                            dynamic_pivot_query = text(dynamic_pivot_query_str)
                            result_proxy_final = connection.execute(dynamic_pivot_query)
                            st.session_state.result_df = pd.DataFrame(result_proxy_final.fetchall(), columns=result_proxy_final.keys())

                            st.success(f"Pivot generated successfully! Found {st.session_state.result_df.shape[0]} rows.")
            except Exception as e:
                st.error(f"An error occurred during pivot generation: {e}")
                st.session_state.result_df = None

    # --- Display Pivot Result ---
    if st.session_state.result_df is not None and not st.session_state.result_df.empty:
        st.markdown("---")
        st.subheader("📈 Final Pivot Result")

        df_to_filter = st.session_state.result_df.copy()
        original_df = st.session_state.result_df

        with st.expander("📊 Filter Results", expanded=True):
            num_filters = len(original_df.columns)
            num_cols = min(num_filters, 4)
            filter_cols = st.columns(num_cols) if num_cols > 0 else [st]
            col_idx = 0

            for column in original_df.columns:
                with filter_cols[col_idx % num_cols]:
                    if pd.api.types.is_object_dtype(original_df[column]) or pd.api.types.is_string_dtype(original_df[column]):
                        unique_values = sorted(original_df[column].dropna().unique())
                        if len(unique_values) < 6:
                            st.markdown(f"**Filter by {column}**")
                            selected_values = [val for val in unique_values if st.checkbox(str(val), value=True, key=f"filter_{column}_{val}")]
                        else:
                            selected_values = st.multiselect(f"Filter by {column}", options=unique_values, default=list(unique_values), key=f"filter_{column}")
                        df_to_filter = df_to_filter[df_to_filter[column].isin(selected_values)]
                    elif pd.api.types.is_numeric_dtype(original_df[column]):
                        min_val = float(original_df[column].min())
                        max_val = float(original_df[column].max())
                        if min_val < max_val:
                            selected_range = st.slider(f"Filter by {column}", min_val, max_val, (min_val, max_val), key=f"filter_{column}")
                            df_to_filter = df_to_filter[df_to_filter[column].between(selected_range[0], selected_range[1])]
                col_idx += 1

        st.info("💡 Select a row and a numeric column to drill down.")

        df_to_filter = df_to_filter.reset_index(drop=True)
        st.dataframe(df_to_filter, use_container_width=True)

        row_to_drill = st.number_input("Enter row number to drill down (1-based index):", min_value=0, max_value=len(df_to_filter), step=1)
        if row_to_drill > 0:
            selected_row_data = df_to_filter.iloc[row_to_drill - 1].to_dict()
            numeric_cols = [col for col in df_to_filter.columns if pd.api.types.is_numeric_dtype(df_to_filter[col])]
            if numeric_cols:
                col_name = st.selectbox("Select a numeric column to drill down:", numeric_cols)
                if col_name:
                    with st.spinner(f"Fetching details for column '{col_name}'..."):
                        drill_df, filter_desc = fetch_drill_down_data(st.session_state.table, st.session_state.pivot_params, selected_row_data, col_name)
                        st.session_state.drill_down_df = drill_df
                        st.session_state.drill_down_info = filter_desc

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

else:
    st.info("Please enter your database credentials and click 'Connect' in the sidebar to begin.")