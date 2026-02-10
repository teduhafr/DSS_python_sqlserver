import React, { useState } from 'react';
import { connectionAPI } from '../services/api';

export default function DatabaseConnection({ onConnect }) {
  const [credentials, setCredentials] = useState({
    server: '',
    database: '',
    username: '',
    password: '',
    port: '',
    dbType: 'mssql', // Default to SQL Server
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleChange = (e) => {
    setCredentials({
      ...credentials,
      [e.target.name]: e.target.value,
    });
    setError('');
    setSuccess('');
  };

  const handleConnect = async (e) => {
    e.preventDefault();
    
    if (!credentials.server || !credentials.database || !credentials.username || !credentials.password) {
      setError('Please fill in all connection details');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await connectionAPI.test(credentials);
      if (response.data.success) {
        setSuccess('Connection successful!');
        onConnect(credentials);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Connection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold text-primary-400 mb-6 flex items-center gap-2">
        <span>🔗</span> Database Connection
      </h2>
      <p className="text-dark-textMuted mb-6">Connect to your Database</p>
      
      <form onSubmit={handleConnect} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Database Type</label>
            <select
              name="dbType"
              value={credentials.dbType}
              onChange={handleChange}
              className="select w-full"
            >
              <option value="mssql">SQL Server</option>
              <option value="postgres">PostgreSQL</option>
            </select>
          </div>
          <div>
            <label className="label">Port</label>
            <input
              type="number"
              name="port"
              value={credentials.port}
              onChange={handleChange}
              placeholder={credentials.dbType === 'postgres' ? '5432' : '1433'}
              className="input w-full"
            />
          </div>
        </div>

        <div>
          <label className="label">Server/Host</label>
          <input
            type="text"
            name="server"
            value={credentials.server}
            onChange={handleChange}
            placeholder="your_server.database.windows.net"
            className="input"
          />
        </div>

        <div>
          <label className="label">Database</label>
          <input
            type="text"
            name="database"
            value={credentials.database}
            onChange={handleChange}
            placeholder="your_database_name"
            className="input"
          />
        </div>

        <div>
          <label className="label">Username</label>
          <input
            type="text"
            name="username"
            value={credentials.username}
            onChange={handleChange}
            placeholder="your_username"
            className="input"
          />
        </div>

        <div>
          <label className="label">Password</label>
          <input
            type="password"
            name="password"
            value={credentials.password}
            onChange={handleChange}
            placeholder="Enter password"
            className="input"
          />
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-700 text-red-200 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        {success && (
          <div className="bg-green-900/30 border border-green-700 text-green-200 px-4 py-3 rounded-lg">
            {success}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="btn-primary w-full"
        >
          {loading ? 'Connecting...' : 'Connect'}
        </button>
      </form>
    </div>
  );
}
