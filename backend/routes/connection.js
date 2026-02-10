import express from 'express';
import { testConnection, closeConnection, testPostgresConnection, executePostgresQuery, closePostgresConnection } from '../config/database.js';

const router = express.Router();

/**
 * @swagger
 * /api/connection/test:
 *   post:
 *     summary: Test database connection
 *     description: Validates database connection credentials (SQL Server or PostgreSQL)
 *     tags: [Connection]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               server:
 *                 type: string
 *               database:
 *                 type: string
 *               username:
 *                 type: string
 *               password:
 *                 type: string
 *               port:
 *                 type: integer
 *               dbType:
 *                 type: string
 *                 enum: [mssql, postgres]
 *                 default: mssql
 *     responses:
 *       200:
 *         description: Connection successful
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *       400:
 *         description: Missing credentials
 *       500:
 *         description: Connection failed
 */
router.post('/test', async (req, res) => {
    const { server, database, username, password, port, dbType } = req.body;

    // For PostgreSQL, 'server' is often referred to as 'host'
    const host = server;

    if (!server || !database || !username || !password) {
        return res.status(400).json({ error: 'All connection fields are required' });
    }

    let client;
    try {
        if (dbType === 'postgres') {
            const pgPort = parseInt(port) || 5432;
            client = await testPostgresConnection({ host, database, username, password, port: pgPort });
            await closePostgresConnection(client);
        } else {
            // Default to MSSQL
            const mssqlPort = parseInt(port) || 1433;
            client = await testConnection({ server, database, username, password, port: mssqlPort });
            await closeConnection(client);
        }
        res.json({ success: true, message: 'Connection successful!' });
    } catch (error) {
        // Ensure connection is closed if it was opened but failed logic (though unlikely for testConnection which throws on fail)
        // For pg client, if new Client() checks connection on connect(), it might throw. 
        if (client) {
            if (dbType === 'postgres') await closePostgresConnection(client);
            else await closeConnection(client);
        }
        res.status(500).json({ error: error.message });
    }
});

/**
 * @swagger
 * /api/connection/tables:
 *   post:
 *     summary: Get list of tables
 *     description: Retrieves all table names from the database
 *     tags: [Connection]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               server:
 *                 type: string
 *               database:
 *                 type: string
 *               username:
 *                 type: string
 *               password:
 *                 type: string
 *               port:
 *                 type: integer
 *               dbType:
 *                 type: string
 *                 enum: [mssql, postgres]
 *     responses:
 *       200:
 *         description: List of tables
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 tables:
 *                   type: array
 *                   items:
 *                     type: string
 *       500:
 *         description: Failed to retrieve tables
 */
router.post('/tables', async (req, res) => {
    const { server, database, username, password, port, dbType } = req.body;
    const host = server;

    let client;
    try {
        let tables = [];

        if (dbType === 'postgres') {
            const pgPort = parseInt(port) || 5432;
            client = await testPostgresConnection({ host, database, username, password, port: pgPort });

            const query = `
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            `;

            const result = await executePostgresQuery(client, query);
            tables = result.rows.map(row => row.table_name);

            await closePostgresConnection(client);
        } else {
            // Default to MSSQL
            client = await testConnection({ server, database, username, password, port: parseInt(port) || 1433 });

            const result = await client.request().query(`
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            `);

            tables = result.recordset.map(row => row.TABLE_NAME);
            await closeConnection(client);
        }

        res.json({ tables });
    } catch (error) {
        if (client) {
            if (dbType === 'postgres') await closePostgresConnection(client);
            else await closeConnection(client);
        }
        res.status(500).json({ error: error.message });
    }
});

export default router;
