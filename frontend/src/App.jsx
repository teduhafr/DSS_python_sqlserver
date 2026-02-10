import React, { useState } from 'react';
import DatabaseConnection from './components/DatabaseConnection';
import PivotControls from './components/PivotControls';
import PivotGrid from './components/PivotGrid';
import DrillDown from './components/DrillDown';
import DataExplorer from './components/DataExplorer';
import { pivotAPI, connectionAPI } from './services/api';

function App() {
  const [credentials, setCredentials] = useState(null);
  const [tables, setTables] = useState([]);
  const [activeTab, setActiveTab] = useState('pivot');
  
  const [pivotData, setPivotData] = useState(null);
  const [pivotParams, setPivotParams] = useState(null);
  const [pivotQuery, setPivotQuery] = useState('');
  
  const [drillDownData, setDrillDownData] = useState(null);
  const [drillDownInfo, setDrillDownInfo] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleConnect = async (creds) => {
    setCredentials(creds);
    try {
      const response = await connectionAPI.getTables(creds);
      setTables(response.data.tables);
    } catch (err) {
      console.error('Failed to fetch tables:', err);
    }
  };

  const handleGeneratePivot = async (params) => {
    setLoading(true);
    setError('');
    setDrillDownData(null);
    setDrillDownInfo(null);
    
    try {
      const response = await pivotAPI.generate(params);
      setPivotData(response.data.data);
      setPivotParams(params);
      setPivotQuery(response.data.query);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate pivot table');
    } finally {
      setLoading(false);
    }
  };

  const handleCellClick = async (rowData, columnName) => {
    if (!pivotParams) return;
    
    setLoading(true);
    try {
      const response = await pivotAPI.drilldown({
        credentials,
        table: pivotParams.table,
        pivotParams,
        clickedRowData: rowData,
        clickedColName: columnName,
      });
      setDrillDownData(response.data.data);
      setDrillDownInfo(response.data.filterInfo);
    } catch (err) {
      console.error('Failed to fetch drill-down data:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <header className="bg-dark-card border-b border-dark-border shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-primary-400">
            📊 Dynamic DSS - Pivot Tables & Data Analysis
          </h1>
          <p className="text-dark-textMuted mt-1">
            Interactive Decision Support System with SQL Server
          </p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            <DatabaseConnection onConnect={handleConnect} />
            
            {credentials && (
              <PivotControls
                credentials={credentials}
                onGenerate={handleGeneratePivot}
              />
            )}
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {credentials ? (
              <>
                {/* Tabs */}
                <div className="card mb-6">
                  <div className="flex gap-2">
                    <button
                      onClick={() => setActiveTab('pivot')}
                      className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeTab === 'pivot'
                          ? 'bg-primary-600 text-white'
                          : 'bg-dark-bg text-dark-textMuted hover:text-dark-text'
                      }`}
                    >
                      ⚡ Pivot Generator
                    </button>
                    <button
                      onClick={() => setActiveTab('explorer')}
                      className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeTab === 'explorer'
                          ? 'bg-primary-600 text-white'
                          : 'bg-dark-bg text-dark-textMuted hover:text-dark-text'
                      }`}
                    >
                      🔍 Data Explorer
                    </button>
                  </div>
                </div>

                {/* Tab Content */}
                {activeTab === 'pivot' && (
                  <div className="space-y-6">
                    {loading && (
                      <div className="card text-center py-12">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto mb-4"></div>
                        <p className="text-dark-textMuted">Generating pivot table...</p>
                      </div>
                    )}

                    {error && (
                      <div className="bg-red-900/30 border border-red-700 text-red-200 px-6 py-4 rounded-lg">
                        {error}
                      </div>
                    )}

                    {pivotQuery && (
                      <div className="card">
                        <h3 className="text-lg font-semibold mb-3">Generated T-SQL Query</h3>
                        <pre className="bg-dark-bg border border-dark-border rounded-lg p-4 text-xs text-dark-textMuted overflow-x-auto">
                          {pivotQuery}
                        </pre>
                      </div>
                    )}

                    {pivotData && pivotParams && (
                      <PivotGrid
                        data={pivotData}
                        pivotParams={pivotParams}
                        onCellClick={handleCellClick}
                      />
                    )}

                    {drillDownData && (
                      <DrillDown
                        data={drillDownData}
                        filterInfo={drillDownInfo}
                        onClose={() => {
                          setDrillDownData(null);
                          setDrillDownInfo(null);
                        }}
                      />
                    )}
                  </div>
                )}

                {activeTab === 'explorer' && (
                  <DataExplorer credentials={credentials} tables={tables} />
                )}
              </>
            ) : (
              <div className="card text-center py-16">
                <div className="text-6xl mb-4">🔌</div>
                <h2 className="text-2xl font-bold mb-2">Not Connected</h2>
                <p className="text-dark-textMuted">
                  Please enter your database credentials and click 'Connect' in the sidebar to begin.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-12 bg-dark-card border-t border-dark-border">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-dark-textMuted text-sm">
          <p>DSS Application v1.0.0 | Backend API available at <a href="http://localhost:5000/api-docs" target="_blank" rel="noopener noreferrer" className="text-primary-400 hover:text-primary-300">http://localhost:5000/api-docs</a></p>
        </div>
      </footer>
    </div>
  );
}

export default App;
