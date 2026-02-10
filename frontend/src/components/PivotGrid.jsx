import React from 'react';
import { AgGridReact } from 'ag-grid-react';

export default function PivotGrid({ data, pivotParams, onCellClick }) {
  if (!data || data.length === 0) {
    return null;
  }

  const columnDefs = Object.keys(data[0]).map(key => ({
    field: key,
    headerName: key,
    sortable: true,
    filter: true,
    resizable: true,
    pinned: pivotParams.rows.includes(key) ? 'left' : null,
  }));

  const gridOptions = {
    defaultColDef: {
      flex: 1,
      minWidth: 100,
    },
    rowSelection: 'single',
    onCellClicked: (params) => {
      if (onCellClick && !pivotParams.rows.includes(params.column.getColId())) {
        onCellClick(params.data, params.column.getColId());
      }
    },
  };

  const title = `${pivotParams.aggFunc} of \`${pivotParams.value}\` by \`${pivotParams.rows.join(', ')}\` across \`${pivotParams.pivotCol}\``;

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4">📈 {title}</h3>
      <div className="ag-theme-alpine-dark" style={{ height: 500, width: '100%' }}>
        <AgGridReact
          rowData={data}
          columnDefs={columnDefs}
          gridOptions={gridOptions}
        />
      </div>
      <p className="text-sm text-dark-textMuted mt-2">
        Found {data.length} rows. Click on a cell to drill down into details.
      </p>
    </div>
  );
}
