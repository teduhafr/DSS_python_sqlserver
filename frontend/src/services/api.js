import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const connectionAPI = {
    test: (credentials) => api.post('/connection/test', credentials),
    getTables: (credentials) => api.post('/connection/tables', credentials),
};

export const metadataAPI = {
    getColumns: (tableName, credentials) =>
        api.post(`/metadata/columns/${tableName}`, credentials),
    getDistinctValues: (tableName, columnName, credentials) =>
        api.post(`/metadata/distinct/${tableName}/${columnName}`, credentials),
};

export const pivotAPI = {
    generate: (params) => api.post('/pivot/generate', params),
    drilldown: (params) => api.post('/pivot/drilldown', params),
};

export const explorerAPI = {
    query: (params) => api.post('/explorer/query', params),
};

export default api;
