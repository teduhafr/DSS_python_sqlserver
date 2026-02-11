import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';

export default function ChartGenerator({ data, pivotParams }) {
  const [chartType, setChartType] = useState('bar');
  const [xAxis, setXAxis] = useState('');
  const [yAxes, setYAxes] = useState([]);
  const [pieLabel, setPieLabel] = useState('');
  const [pieValue, setPieValue] = useState('');

  // Extract available columns for axes
  const { rowCols, valueCols, validData } = useMemo(() => {
    if (!data || !pivotParams) return { rowCols: [], valueCols: [], validData: [] };

    const rowCols = pivotParams.rows || [];
    
    // Filter out 'Total' row if it exists and is enabled
    let filteredData = [...data];
    if (pivotParams.showRowTotals && rowCols.length > 0) {
       // Check if the last row is a Total row. 
       // In the dss.py it checks if the first row column value is 'Total'.
       // We'll filter based on that heuristic.
       const firstRowCol = rowCols[0];
       filteredData = filteredData.filter(row => row[firstRowCol] !== 'Total');
    }

    // Identify value columns (dynamic columns created by pivot + maybe others)
    // Basically all keys that are NOT in rowCols
    const allKeys = Object.keys(data[0] || {});
    const valueCols = allKeys.filter(key => !rowCols.includes(key));

    return { rowCols, valueCols, validData: filteredData };
  }, [data, pivotParams]);

  // Set defaults when options change
  React.useEffect(() => {
    if (rowCols.length > 0 && !xAxis) setXAxis(rowCols[0]);
    if (valueCols.length > 0 && yAxes.length === 0) setYAxes([valueCols[0]]);
    if (rowCols.length > 0 && !pieLabel) setPieLabel(rowCols[0]);
    if (valueCols.length > 0 && !pieValue) setPieValue(valueCols[0]);
  }, [rowCols, valueCols]);

  if (!data || data.length === 0) return null;

  const generateChartData = () => {
    if (chartType === 'pie') {
      if (!pieLabel || !pieValue) return [];
      return [{
        labels: validData.map(d => d[pieLabel]),
        values: validData.map(d => d[pieValue]),
        type: 'pie',
        textinfo: 'label+percent',
        hoverinfo: 'label+value+percent'
      }];
    } else {
      // Bar, Line, Scatter
      if (!xAxis || yAxes.length === 0) return [];
      
      return yAxes.map(yCol => ({
        x: validData.map(d => d[xAxis]),
        y: validData.map(d => d[yCol]),
        type: chartType === 'scatter' ? 'scatter' : chartType, // plotly uses 'scatter' for both line and scatter, differentiated by mode
        mode: chartType === 'scatter' ? 'markers' : (chartType === 'line' ? 'lines+markers' : undefined),
        name: yCol
      }));
    }
  };

  const layout = {
    title: chartType === 'pie' 
      ? `${pieValue} by ${pieLabel}` 
      : `${yAxes.join(', ')} by ${xAxis}`,
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      color: '#e2e8f0' // match dark theme text
    },
    xaxis: {
      title: xAxis,
      gridcolor: '#334155'
    },
    yaxis: {
      gridcolor: '#334155'
    },
    margin: { t: 50, r: 20, l: 40, b: 50 },
    legend: {
        orientation: 'h',
        y: -0.2
    }
  };

  return (
    <div className="card mt-6">
      <h3 className="text-lg font-semibold mb-4 text-primary-400">📊 Chart Generator</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="label">Chart Type</label>
          <select 
            value={chartType} 
            onChange={e => setChartType(e.target.value)}
            className="select"
          >
            <option value="bar">Bar Chart</option>
            <option value="line">Line Chart</option>
            <option value="scatter">Scatter Plot</option>
            <option value="pie">Pie Chart</option>
          </select>
        </div>

        {chartType !== 'pie' ? (
          <>
            <div>
              <label className="label">X-Axis</label>
              <select 
                value={xAxis} 
                onChange={e => setXAxis(e.target.value)}
                className="select"
              >
                {rowCols.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            <div className="md:col-span-2">
              <label className="label">Y-Axis (hold Ctrl to select multiple)</label>
              <select 
                multiple
                value={yAxes} 
                onChange={e => setYAxes(Array.from(e.target.selectedOptions, o => o.value))}
                className="select h-10" // slightly taller? standard inputs are usually single line.
                style={{height: '2.5rem'}}
              >
                {valueCols.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          </>
        ) : (
          <>
             <div>
              <label className="label">Labels</label>
              <select 
                value={pieLabel} 
                onChange={e => setPieLabel(e.target.value)}
                className="select"
              >
                {rowCols.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Values</label>
              <select 
                value={pieValue} 
                onChange={e => setPieValue(e.target.value)}
                className="select"
              >
                {valueCols.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          </>
        )}
      </div>

      <div className="w-full h-[500px] border border-dark-border rounded-lg p-2 bg-dark-bg">
        <Plot
          data={generateChartData()}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
          config={{responsive: true}}
        />
      </div>
    </div>
  );
}
