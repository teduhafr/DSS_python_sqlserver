import React, { useState, useEffect } from 'react';
import { connectionAPI, metadataAPI } from '../services/api';

export default function PivotControls({ credentials, onGenerate }) {
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [columns, setColumns] = useState([]);
  const [numericColumns, setNumericColumns] = useState([]);
  
  const [rows, setRows] = useState([]);
  const [pivotCol, setPivotCol] = useState('');
  const [value, setValue] = useState('');
  const [aggFunc, setAggFunc] = useState('SUM');
  const [showRowTotals, setShowRowTotals] = useState(false);
  const [showColTotals, setShowColTotals] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (credentials) {
      fetchTables();
    }
  }, [credentials]);

  useEffect(() => {
    if (selectedTable && credentials) {
      fetchColumns();
    }
  }, [selectedTable]);

  const fetchTables = async () => {
    try {
      const response = await connectionAPI.getTables(credentials);
      setTables(response.data.tables);
    } catch (err) {
      setError('Failed to fetch tables');
    }
  };

  const fetchColumns = async () => {
    try {
      const response = await metadataAPI.getColumns(selectedTable, credentials);
      setColumns(response.data.columns);
      setNumericColumns(response.data.numericColumns);
      
      if (response.data.columns.length > 0) {
        setRows([response.data.columns[0]]);
        setPivotCol(response.data.columns.length > 1 ? response.data.columns[1] : response.data.columns[0]);
        setValue(response.data.columns.length > 2 ? response.data.columns[2] : response.data.columns[0]);
      }
    } catch (err) {
      setError('Failed to fetch columns');
    }
  };

  const handleGenerate = () => {
    if (!selectedTable || rows.length === 0 || !pivotCol || !value || !aggFunc) {
      setError('Please select all required fields');
      return;
    }

    const isNumericAgg = ['SUM', 'AVG'].includes(aggFunc);
    if (isNumericAgg && !numericColumns.includes(value)) {
      setError(`Aggregation '${aggFunc}' requires a numeric value column`);
      return;
    }

    setError('');
    onGenerate({
      credentials,
      table: selectedTable,
      rows,
      pivotCol,
      value,
      aggFunc,
      showRowTotals,
      showColTotals,
    });
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-primary-400 mb-4">⚙️ Pivot Controls</h2>
      
      <div className="space-y-4">
        <div>
          <label className="label">Select Table</label>
          <select
            value={selectedTable}
            onChange={(e) => setSelectedTable(e.target.value)}
            className="select"
          >
            <option value="">-- Select a table --</option>
            {tables.map(table => (
              <option key={table} value={table}>{table}</option>
            ))}
          </select>
        </div>

        {columns.length > 0 && (
          <>
            <div>
              <label className="label">Rows</label>
              <select
                multiple
                value={rows}
                onChange={(e) => setRows(Array.from(e.target.selectedOptions, option => option.value))}
                className="select h-24"
              >
                {columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <p className="text-xs text-dark-textMuted mt-1">Hold Ctrl/Cmd to select multiple</p>
            </div>

            <div>
              <label className="label">Columns</label>
              <select
                value={pivotCol}
                onChange={(e) => setPivotCol(e.target.value)}
                className="select"
              >
                {columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="label">Value</label>
              <select
                value={value}
                onChange={(e) => setValue(e.target.value)}
                className="select"
              >
                {columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="label">Aggregation</label>
              <select
                value={aggFunc}
                onChange={(e) => setAggFunc(e.target.value)}
                className="select"
              >
                <option value="SUM">SUM</option>
                <option value="AVG">AVG</option>
                <option value="COUNT">COUNT</option>
                <option value="MAX">MAX</option>
                <option value="MIN">MIN</option>
              </select>
            </div>

            <div className="border-t border-dark-border pt-4">
              <p className="label mb-3">📊 Totals</p>
              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showRowTotals}
                    onChange={(e) => setShowRowTotals(e.target.checked)}
                    className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                  />
                  <span className="text-dark-text">Show Row Totals (horizontal)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showColTotals}
                    onChange={(e) => setShowColTotals(e.target.checked)}
                    className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                  />
                  <span className="text-dark-text">Show Column Totals (vertical)</span>
                </label>
              </div>
            </div>

            {error && (
              <div className="bg-red-900/30 border border-red-700 text-red-200 px-4 py-3 rounded-lg text-sm">
                {error}
              </div>
            )}

            <button
              onClick={handleGenerate}
              disabled={loading}
              className="btn-primary w-full"
            >
              🚀 Generate Pivot Table
            </button>
          </>
        )}
      </div>
    </div>
  );
}
