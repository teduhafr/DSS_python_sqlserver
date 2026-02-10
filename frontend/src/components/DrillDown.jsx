import React from 'react';
import { AgGridReact } from 'ag-grid-react';

export default function DrillDown({ data, filterInfo, onClose }) {
  if (!data) return null;

  const columnDefs = Object.keys(data[0] || {}).map(key => ({
    field: key,
    headerName: key,
    sortable: true,
    filter: true,
    resizable: true,
  }));

  return (
    <div className="card mt-6 animate-slide-up">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">🔬 Drill-Down Details</h3>
        <button onClick={onClose} className="btn-secondary text-sm">
          Hide Details
        </button>
      </div>

      {filterInfo && (
        <div className="bg-dark-bg border border-dark-border rounded-lg p-4 mb-4">
          <p className="text-sm font-medium mb-2">Showing raw data for:</p>
          <pre className="text-xs text-dark-textMuted">
            {JSON.stringify(filterInfo, null, 2)}
          </pre>
        </div>
      )}

      {data && data.length > 0 ? (
        <>
          <div className="ag-theme-alpine-dark" style={{ height: 400, width: '100%' }}>
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
            Displaying {data.length} raw data rows
          </p>
        </>
      ) : (
        <div className="bg-yellow-900/30 border border-yellow-700 text-yellow-200 px-4 py-3 rounded-lg">
          No underlying data found for the selected cell
        </div>
      )}
    </div>
  );
}
