import express from 'express';
import { testConnection, closeConnection, testPostgresConnection, executePostgresQuery, closePostgresConnection } from '../config/database.js';

const router = express.Router();

/**
 * @swagger
 * /api/explorer/query:
 *   post:
 *     summary: Query and filter data
 *     description: Retrieve data from a table with optional filters
 *     tags: [Data Explorer]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/FilterParams'
 *     responses:
 *       200:
 *         description: Filtered data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                 count:
 *                   type: integer
 *       500:
 *         description: Failed to query data
 */
router.post('/query', async (req, res) => {
    const { credentials, table, filters = [] } = req.body;

    let client;
    try {
        if (credentials.dbType === 'postgres') {
            const pgPort = parseInt(credentials.port) || 5432;
            client = await testPostgresConnection({ ...credentials, port: pgPort });

            let query = `SELECT * FROM "${table}"`;
            const whereClauses = [];
            const values = []; // For parameterized queries if we were using them, but here we build string

            filters.forEach((filter) => {
                const { column, type, value } = filter;

                // Basic sanitization/logic for Postgres
                switch (type) {
                    case 'text_contains':
                        whereClauses.push(`"${column}" LIKE '%${value.replace(/'/g, "''")}%'`);
                        break;
                    case 'numeric_gte':
                        whereClauses.push(`"${column}" >= ${value}`);
                        break;
                    case 'numeric_lte':
                        whereClauses.push(`"${column}" <= ${value}`);
                        break;
                    case 'numeric_range':
                        whereClauses.push(`"${column}" BETWEEN ${value[0]} AND ${value[1]}`);
                        break;
                }
            });

            if (whereClauses.length > 0) {
                query += ` WHERE ${whereClauses.join(' AND ')}`;
            }

            // Limit results to prevent massive loads
            query += ' LIMIT 1000';

            const result = await executePostgresQuery(client, query);

            await closePostgresConnection(client);
            res.json({
                data: result.rows,
                count: result.rows.length
            });

        } else {
            // Default to MSSQL
            if (!credentials.port) {
                credentials.port = 1433;
            }

            client = await testConnection(credentials);

            let query = `SELECT TOP 1000 * FROM [${table}]`; // Added TOP 1000 for safety
            const whereClauses = [];

            filters.forEach((filter) => {
                const { column, type, value } = filter;

                switch (type) {
                    case 'text_contains':
                        whereClauses.push(`[${column}] LIKE '%${value.replace(/'/g, "''")}%'`);
                        break;
                    case 'numeric_gte':
                        whereClauses.push(`[${column}] >= ${value}`);
                        break;
                    case 'numeric_lte':
                        whereClauses.push(`[${column}] <= ${value}`);
                        break;
                    case 'numeric_range':
                        whereClauses.push(`[${column}] BETWEEN ${value[0]} AND ${value[1]}`);
                        break;
                }
            });

            if (whereClauses.length > 0) {
                query += ` WHERE ${whereClauses.join(' AND ')}`;
            }

            const result = await client.request().query(query);

            await closeConnection(client);
            res.json({
                data: result.recordset,
                count: result.recordset.length
            });
        }
    } catch (error) {
        if (client) {
            if (credentials.dbType === 'postgres') await closePostgresConnection(client);
            else await closeConnection(client);
        }
        res.status(500).json({ error: error.message });
    }
});

export default router;
