import sql from 'mssql';
import { Client } from 'pg';

/**
 * Test database connection with provided credentials
 * @param {Object} credentials - Database connection credentials
 * @returns {Promise<sql.ConnectionPool>} - Connection pool
 */
export async function testConnection(credentials) {
    const { server, database, username, password, port } = credentials;

    const config = {
        server,
        database,
        user: username,
        password,
        port: parseInt(port) || 1433, // Default SQL Server port
        options: {
            encrypt: true, // Use encryption for Azure SQL
            trustServerCertificate: true, // Set to true for local dev/self-signed certs
            connectTimeout: 10000,
        },
        pool: {
            max: 10,
            min: 0,
            idleTimeoutMillis: 30000,
        },
    };

    try {
        const pool = await sql.connect(config);
        return pool;
    } catch (error) {
        throw new Error(`Database connection failed: ${error.message}`);
    }
}

/**
 * Execute a query with parameters
 * @param {sql.ConnectionPool} pool - Connection pool
 * @param {string} query - SQL query string
 * @param {Object} params - Query parameters
 * @returns {Promise<sql.IResult>} - Query result
 */
export async function executeQuery(pool, query, params = {}) {
    try {
        const request = pool.request();

        // Add parameters to the request
        for (const [key, value] of Object.entries(params)) {
            request.input(key, value);
        }

        const result = await request.query(query);
        return result;
    } catch (error) {
        throw new Error(`Query execution failed: ${error.message}`);
    }
}

/**
 * Close database connection pool
 * @param {sql.ConnectionPool} pool - Connection pool to close
 */
export async function closeConnection(pool) {
    if (pool) {
        await pool.close();
    }
}

/**
 * Test PostgreSQL database connection with provided credentials
 * @param {Object} credentials - Database connection credentials
 * @returns {Promise<Client>} - PostgreSQL client instance
 */
export async function testPostgresConnection(credentials) {
    const { host, database, username, password, port } = credentials;

    const client = new Client({
        host,
        database,
        user: username,
        password,
        port: parseInt(port) || 5432, // Default PostgreSQL port
    });

    try {
        await client.connect();
        return client;
    } catch (error) {
        throw new Error(`PostgreSQL connection failed: ${error.message}`);
    }
}

/**
 * Execute a query with parameters for PostgreSQL
 * @param {Client} client - PostgreSQL client instance
 * @param {string} query - SQL query string
 * @param {Array} params - Query parameters
 * @returns {Promise<Object>} - Query result
 */
export async function executePostgresQuery(client, query, params = []) {
    try {
        const result = await client.query(query, params);
        return result;
    } catch (error) {
        throw new Error(`PostgreSQL query execution failed: ${error.message}`);
    }
}

/**
 * Close PostgreSQL database connection
 * @param {Client} client - PostgreSQL client instance
 */
export async function closePostgresConnection(client) {
    if (client) {
        await client.end();
    }
}
