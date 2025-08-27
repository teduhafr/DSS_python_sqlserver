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

# --- Database Connection Function ---
def get_sql_server_engine(server, database, username, password):
    """Creates a SQLAlchemy engine from user inputs and tests the connection."""
    try:
        driver = "ODBC Driver 17 for SQL Server"
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
        engine = create_engine(conn_str, connect_args={"timeout": 10})
        # Test connection
        connection = engine.connect()
        connection.close()
        return engine
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {e}")
        return None

@st.dialog("Drill-Down Details")
def show_drill_down_data(table_name, pivot_params, clicked_row_data, clicked_col_name):
    """Constructs and runs a query to get the raw data for a pivot cell."""
    st.write(f"Showing raw data for:")
    # Create a neat display of the filters being applied
    filter_desc = {**{col: clicked_row_data[col] for col in pivot_params['rows']}, pivot_params['pivot_col']: clicked_col_name}
    st.json(filter_desc)

    with st.spinner("Fetching details..."):
        try:
            where_clauses = []
            params = {}

            # Add conditions for the pivot rows (e.g., Country = 'USA')
            for col in pivot_params['rows']:
                param_name = f"param_{col.replace(' ', '_')}"
                where_clauses.append(f'"{col}" = :{param_name}')
                params[param_name] = clicked_row_data[col]

            # Add condition for the pivot column (e.g., Year = 2023)
            pivot_col = pivot_params['pivot_col']
            param_name = f"param_{pivot_col.replace(' ', '_')}"
            where_clauses.append(f'"{pivot_col}" = :{param_name}')
            params[param_name] = clicked_col_name

            # Construct and run the query
            query_str = f'SELECT * FROM "{table_name}" WHERE {" AND ".join(where_clauses)}'
            query = text(query_str)

            with st.session_state.engine.connect() as connection:
                drill_df = pd.read_sql(query, connection, params=params)
            
            st.dataframe(drill_df, use_container_width=True)
            st.info(f"Displaying {len(drill_df)} raw data rows.")

        except Exception as e:
            st.error(f"Could not fetch drill-down data: {e}")

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
        # Store previous table to detect changes
        prev_table = st.session_state.table
        st.session_state.table = st.selectbox("Select Table", options=st.session_state.tables)

        # If table has changed, clear the previous pivot result to avoid confusion
        if prev_table != st.session_state.table:
            st.session_state.result_df = None
            st.session_state.preview_df = None

        if st.session_state.table:
            try:
                with st.session_state.engine.connect() as connection:
                    # This is fine since it's a simple, non-dynamic query string
                    sample_df = pd.read_sql(f'SELECT TOP 1 * FROM "{st.session_state.table}"', connection)
                    st.session_state.available_columns = sample_df.columns.tolist()
                    # Get column types to identify searchable columns for the preview
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
            # Use a copy to avoid modifying session state directly in widget defaults
            cols_options = st.session_state.available_columns[:]
            rows = st.multiselect("Rows", options=cols_options, default=cols_options[0] if cols_options else None)
            pivot_col = st.selectbox("Columns", options=cols_options, index=1 if len(cols_options) > 1 else 0)
            value = st.selectbox("Value", options=cols_options, index=2 if len(cols_options) > 2 else 0)
            agg_func = st.selectbox("Aggregation", options=['SUM', 'AVG', 'COUNT', 'MAX', 'MIN'])

    st.title("⚡ Dynamic Pivot Table Generator")

    # === NEW SECTION START ===
    # Section to display original data
    if st.session_state.table: # Only show this if a table is selected
        with st.expander("📄 View Original Data Preview"):
            # --- UI Controls are now all defined before the button ---
            rows_to_preview = st.number_input(
                "Number of rows to preview",
                min_value=10,
                max_value=100000,
                value=100,
                step=10
            )

            st.markdown("##### 🔎 Search Before Loading")
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                column_to_search = st.selectbox(
                    "Search in column",
                    options=["All Text Columns"] + st.session_state.get('searchable_columns', []),
                    key="preview_search_col"
                )
            with col2:
                search_term = st.text_input(
                    "Enter search term (optional)",
                    key="preview_search_term",
                    placeholder="Case-insensitive search..."
                )

            if st.button("Load Preview Data"):
                st.session_state.preview_df = None # Clear previous preview data
                try:
                    spinner_msg = f"Loading top {rows_to_preview} rows from '{st.session_state.table}'..."
                    if search_term:
                        spinner_msg = f"Searching for '{search_term}' and loading top {rows_to_preview} rows..."

                    with st.spinner(spinner_msg):
                        with st.session_state.engine.connect() as connection:
                            # Build the query with server-side filtering
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
                            # Store in session state to enable pagination without re-querying
                            st.session_state.preview_df = pd.read_sql(query, connection, params=params)
                except Exception as e:
                    st.error(f"Failed to load original data: {e}")
                    st.session_state.preview_df = None

            # Display logic with pagination if data is loaded
            if st.session_state.preview_df is not None and not st.session_state.preview_df.empty:
                st.success(f"Found and loaded {len(st.session_state.preview_df)} rows.")
                df_for_display = st.session_state.preview_df # No client-side filtering needed
                
                # --- Pagination logic now works on the filtered dataframe ---
                total_rows = len(df_for_display)
                
                # Pagination controls
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
                    # Use markdown with vertical space for better alignment with number_input boxes
                    st.markdown("<br>", unsafe_allow_html=True) 
                    info_text = f"Showing rows {start_idx+1}-{min(end_idx, total_rows)} of {total_rows} found"
                    st.info(info_text)

                st.dataframe(df_to_display, use_container_width=True)
    
    st.markdown("---") # Optional: Adds a nice separator line
    # === NEW SECTION END ===

    if st.button("🚀 Generate Pivot Table"):
        if not all([rows, pivot_col, value, st.session_state.table]):
            st.warning("Please select a table, rows, a column, and a value.")
        else:
            try:
                # Clear previous results before generating new ones
                st.session_state.pivot_params = {
                    "rows": rows,
                    "pivot_col": pivot_col,
                    "value": value,
                    "agg_func": agg_func
                }
                st.session_state.result_df = None
                with st.session_state.engine.connect() as connection:
                    # Step 1: Get unique pivot values
                    distinct_query = text(f"""
                        SELECT DISTINCT "{pivot_col}"
                        FROM "{st.session_state.table}"
                        WHERE "{pivot_col}" IS NOT NULL
                        ORDER BY "{pivot_col}";
                    """)
                    
                    result_proxy = connection.execute(distinct_query)
                    distinct_values_df = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys())
                    
                    pivot_values = [str(val) for val in distinct_values_df[pivot_col].tolist()]

                    if not pivot_values:
                        st.warning(f"No distinct values found for column '{pivot_col}'. Pivot cannot be generated.")
                    else:
                        pivot_in_clause = ", ".join([f'"{v}"' for v in pivot_values])
                        row_cols_str = ", ".join([f'"{r}"' for r in rows])

                        # Step 2: Build dynamic pivot query
                        dynamic_pivot_query_str = f"""
                            SELECT {row_cols_str}, {pivot_in_clause}
                            FROM (
                                SELECT {row_cols_str}, "{pivot_col}", "{value}"
                                FROM "{st.session_state.table}"
                            ) AS SourceTable
                            PIVOT (
                                {agg_func}("{value}")
                                FOR "{pivot_col}" IN ({pivot_in_clause})
                            ) AS PivotTable;
                        """
                        dynamic_pivot_query = text(dynamic_pivot_query_str)
                        
                        st.subheader("Generated T-SQL Query")
                        st.code(dynamic_pivot_query_str, language="sql")
                        
                        result_proxy_final = connection.execute(dynamic_pivot_query)
                        # Store result in session state to enable filtering
                        st.session_state.result_df = pd.DataFrame(result_proxy_final.fetchall(), columns=result_proxy_final.keys())
                        st.success(f"Pivot generated successfully! Found {st.session_state.result_df.shape[0]} rows.")

            except Exception as e:
                st.error(f"An error occurred during pivot generation: {e}")
                st.session_state.result_df = None # Ensure no stale data is shown on error

    # --- Display and Filter Section ---
    if st.session_state.result_df is not None and not st.session_state.result_df.empty:
        st.markdown("---")
        st.subheader("📈 Final Pivot Result")
 
        # Use a copy for filtering, and the original for populating filter options
        df_to_filter = st.session_state.result_df.copy()
        original_df = st.session_state.result_df
 
        with st.expander("📊 Filter Results", expanded=True):
            # Use columns for a cleaner layout, max 4 columns
            num_filters = len(original_df.columns)
            num_cols = min(num_filters, 4)
            filter_cols = st.columns(num_cols) if num_cols > 0 else [st]
            col_idx = 0
 
            for column in original_df.columns:
                with filter_cols[col_idx % num_cols]:
                    # For categorical columns (object or string)
                    if pd.api.types.is_object_dtype(original_df[column]) or pd.api.types.is_string_dtype(original_df[column]):
                        unique_values = sorted(original_df[column].dropna().unique())
                        
                        # Heuristic: if few unique values (<6), use checkboxes. Otherwise, use a multiselect dropdown.
                        if len(unique_values) < 6:
                            st.markdown(f"**Filter by {column}**")
                            selected_values = [
                                val for val in unique_values 
                                if st.checkbox(str(val), value=True, key=f"filter_{column}_{val}")
                            ]
                        else:
                            selected_values = st.multiselect(
                                f"Filter by {column}",
                                options=unique_values,
                                default=list(unique_values),
                                key=f"filter_{column}"
                            )
                        df_to_filter = df_to_filter[df_to_filter[column].isin(selected_values)]
 
                    # For numerical columns
                    elif pd.api.types.is_numeric_dtype(original_df[column]):
                        min_val = float(original_df[column].min())
                        max_val = float(original_df[column].max())
                        if min_val < max_val:
                            selected_range = st.slider(f"Filter by {column}", min_val, max_val, (min_val, max_val), key=f"filter_{column}")
                            df_to_filter = df_to_filter[df_to_filter[column].between(selected_range[0], selected_range[1])]
                col_idx += 1
 
        # Display the filtered pivot table
        st.dataframe(df_to_filter, use_container_width=True)
        st.info(f"Displaying {df_to_filter.shape[0]} of {original_df.shape[0]} total rows.")

        # --- Drill-Down Controls (replaces AgGrid) ---
        st.markdown("---")
        st.markdown("#### 🔬 Drill-Down into Data")
        st.info("💡 Select a row and column from the table above, then click the button to see the underlying raw data.")

        if not df_to_filter.empty:
            row_identifier_cols = st.session_state.pivot_params.get('rows', [])
            
            # Create a temporary, human-readable column for the row selection dropdown
            # This handles multi-column rows gracefully (e.g., "USA - Electronics")
            temp_df = df_to_filter.copy()
            temp_df['__display_row__'] = temp_df[row_identifier_cols].astype(str).agg(' - '.join, axis=1)

            col1, col2 = st.columns(2)
            with col1:
                selected_row_display = st.selectbox(
                    "Select Row",
                    options=temp_df['__display_row__'],
                    key="drill_down_row"
                )
            
            # The columns we can drill into are the numeric ones that are NOT part of the row identifiers
            drillable_cols = [
                col for col in df_to_filter.columns 
                if pd.api.types.is_numeric_dtype(df_to_filter[col]) and col not in row_identifier_cols
            ]
            
            with col2:
                selected_col_to_drill = st.selectbox(
                    "Select Column",
                    options=drillable_cols,
                    key="drill_down_col"
                )

            if st.button("Show Details"):
                if selected_row_display and selected_col_to_drill:
                    # Find the full row data corresponding to the user's selection
                    clicked_row_data = temp_df[temp_df['__display_row__'] == selected_row_display].iloc[0].to_dict()
                    
                    # Call the existing dialog function, which handles the DB query
                    show_drill_down_data(st.session_state.table, st.session_state.pivot_params, clicked_row_data, selected_col_to_drill)
        else:
            st.warning("The pivot table is empty. Nothing to drill down into.")

else:
    st.info("Please enter your database credentials and click 'Connect' in the sidebar to begin.")