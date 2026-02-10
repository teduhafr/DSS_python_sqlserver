import React, { useState } from 'react';
import { explorerAPI, metadataAPI } from '../services/api';
import { AgGridReact } from 'ag-grid-react';

export default function DataExplorer({ credentials, tables }) {
  const [selectedTable, setSelectedTable] = useState('');
  const [columns, setColumns] = useState([]);
  const [searchableColumns, setSearchableColumns] = useState([]);
  const [numericColumns, setNumericColumns] = useState([]);
  const [filters, setFilters] = useState([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTableChange = async (table) => {
    setSelectedTable(table);
    setData(null);
    setFilters([]);

    if (table && credentials) {
      try {
        const response = await metadataAPI.getColumns(table, credentials);
        setColumns(response.data.columns);
        setSearchableColumns(response.data.searchableColumns);
        setNumericColumns(response.data.numericColumns);
      } catch (err) {
        console.error('Failed to fetch columns:', err);
      }
    }
  };

  const addFilter = () => {
    setFilters([...filters, { column: '', type: '', value: '' }]);
  };

  const updateFilter = (index, field, value) => {
    const newFilters = [...filters];
    newFilters[index][field] = value;
    setFilters(newFilters);
  };

  const removeFilter = (index) => {
    setFilters(filters.filter((_, i) => i !== index));
  };

  const handleQuery = async () => {
    if (!selectedTable) return;

    setLoading(true);
    try {
      const validFilters = filters.filter(f => f.column && f.type && f.value);
      const response = await explorerAPI.query({
        credentials,
        table: selectedTable,
        filters: validFilters,
      });
      setData(response.data.data);
    } catch (err) {
      console.error('Failed to query data:', err);
    } finally {
      setLoading(false);
    }
  };

  const columnDefs = data && data.length > 0 
    ? Object.keys(data[0]).map(key => ({
        field: key,
        headerName: key,
        sortable: true,
        filter: true,
        resizable: true,
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-2xl font-bold text-primary-400 mb-6">🔍 Data Explorer</h2>

        <div className="space-y-4">
          <div>
            <label className="label">Select Table</label>
            <select
              value={selectedTable}
              onChange={(e) => handleTableChange(e.target.value)}
              className="select"
            >
              <option value="">-- Select a table --</option>
              {tables.map(table => (
                <option key={table} value={table}>{table}</option>
              ))}
            </select>
          </div>

          {selectedTable && (
            <>
              <div className="border-t border-dark-border pt-4">
                <div className="flex justify-between items-center mb-3">
                  <label className="label mb-0">Filters (combined with AND)</label>
                  <button
                    onClick={addFilter}
                    className="btn-secondary text-sm"
                    disabled={filters.length >= 5}
                  >
                    + Add Filter
                  </button>
                </div>

                {filters.map((filter, index) => (
                  <div key={index} className="flex gap-2 mb-2 items-end">
                    <div className="flex-1">
                      <select
                        value={filter.column}
                        onChange={(e) => updateFilter(index, 'column', e.target.value)}
                        className="select"
                      >
                        <option value="">-- Select column --</option>
                        {[...searchableColumns, ...numericColumns].map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>

                    {filter.column && (
                      <>
                        {numericColumns.includes(filter.column) ? (
                          <div className="flex-1 flex gap-2">
                            <input
                              type="number"
                              placeholder="From"
                              value={filter.value[0] || ''}
                              onChange={(e) => updateFilter(index, 'value', [e.target.value, filter.value[1] || ''])}
                              className="input"
                            />
                            <input
                              type="number"
                              placeholder="To"
                              value={filter.value[1] || ''}
                              onChange={(e) => {
                                const val = [filter.value[0] || '', e.target.value];
                                updateFilter(index, 'type', val[0] && val[1] ? 'numeric_range' : val[0] ? 'numeric_gte' : 'numeric_lte');
                                updateFilter(index, 'value', val);
                              }}
                              className="input"
                            />
                          </div>
                        ) : (
                          <div className="flex-1">
                            <input
                              type="text"
                              placeholder="Contains value..."
                              value={filter.value}
                              onChange={(e) => {
                                updateFilter(index, 'type', 'text_contains');
                                updateFilter(index, 'value', e.target.value);
                              }}
                              className="input"
                            />
                          </div>
                        )}
                      </>
                    )}

                    <button
                      onClick={() => removeFilter(index)}
                      className="btn-secondary px-3"
                      title="Remove filter"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>

              <button
                onClick={handleQuery}
                disabled={loading}
                className="btn-primary w-full"
              >
                {loading ? 'Loading...' : 'Show Data / Apply Filters'}
              </button>
            </>
          )}
        </div>
      </div>

      {data && data.length > 0 && (
        <div className="card animate-fade-in">
          <h3 className="text-lg font-semibold mb-4">Results</h3>
          <div className="ag-theme-alpine-dark" style={{ height: 500, width: '100%' }}>
            <AgGridReact
              rowData={data}
              columnDefs={columnDefs}
              defaultColDef={{
                flex: 1,
                minWidth: 100,
              }}
            />
          </div>
          <p className="text-sm text-dark-textMuted mt-2">
            Showing {data.length} rows
          </p>
        </div>
      )}
    </div>
  );
}
