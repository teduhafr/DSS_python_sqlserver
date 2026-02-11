import express from 'express';
import { testConnection, closeConnection, testPostgresConnection, executePostgresQuery, closePostgresConnection } from '../config/database.js';

const router = express.Router();

/**
 * @swagger
 * /api/metadata/columns/{tableName}:
 *   post:
 *     summary: Get column metadata for a table
 *     description: Returns column information including data types and whether they are searchable or numeric
 *     tags: [Metadata]
 *     parameters:
 *       - in: path
 *         name: tableName
 *         required: true
 *         schema:
 *           type: string
 *         description: Name of the table
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ConnectionCredentials'
 *     responses:
 *       200:
 *         description: Column metadata
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 columns:
 *                   type: array
 *                   items:
 *                     type: string
 *                 searchableColumns:
 *                   type: array
 *                   items:
 *                     type: string
 *                 numericColumns:
 *                   type: array
 *                   items:
 *                     type: string
 *       500:
 *         description: Failed to retrieve column metadata
 */
router.post('/columns/:tableName', async (req, res) => {
    const { tableName } = req.params;
    const { server, database, username, password, port, dbType } = req.body;

    let client;
    try {
        if (dbType === 'postgres') {
            const pgPort = parseInt(port) || 5432;
            client = await testPostgresConnection({ host: server, database, username, password, port: pgPort });

            const query = `
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '${tableName}'
                ORDER BY ordinal_position;
            `;

            const result = await executePostgresQuery(client, query);

            const columns = [];
            const searchableColumns = [];
            const numericColumns = [];

            result.rows.forEach(col => {
                columns.push(col.column_name);
                const dataType = col.data_type.toLowerCase();

                if (dataType.includes('char') || dataType.includes('text')) {
                    searchableColumns.push(col.column_name);
                }

                if (['int', 'float', 'decimal', 'numeric', 'money', 'bigint', 'smallint', 'tinyint', 'real', 'double precision'].some(t => dataType.includes(t))) {
                    numericColumns.push(col.column_name);
                }
            });

            await closePostgresConnection(client);
            res.json({ columns, searchableColumns, numericColumns });

        } else {
            // Default to MSSQL
            const mssqlPort = parseInt(port) || 1433;
            client = await testConnection({ server, database, username, password, port: mssqlPort });

            // Get column information from schema
            const columnInfoResult = await client.request().query(`
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '${tableName}'
                ORDER BY ORDINAL_POSITION
            `);

            const columns = [];
            const searchableColumns = [];
            const numericColumns = [];

            columnInfoResult.recordset.forEach(col => {
                columns.push(col.COLUMN_NAME);
                const dataType = col.DATA_TYPE.toLowerCase();

                if (dataType.includes('char') || dataType.includes('text')) {
                    searchableColumns.push(col.COLUMN_NAME);
                }

                if (['int', 'float', 'decimal', 'numeric', 'money', 'bigint', 'smallint', 'tinyint', 'real'].some(t => dataType.includes(t))) {
                    numericColumns.push(col.COLUMN_NAME);
                }
            });

            await closeConnection(client);
            res.json({ columns, searchableColumns, numericColumns });
        }
    } catch (error) {
        if (client) {
            if (dbType === 'postgres') await closePostgresConnection(client);
            else await closeConnection(client);
        }
        res.status(500).json({ error: error.message });
    }
});

/**
 * @swagger
 * /api/metadata/distinct/{tableName}/{columnName}:
 *   post:
 *     summary: Get distinct values for a column
 *     description: Returns all unique values in a specific column
 *     tags: [Metadata]
 *     parameters:
 *       - in: path
 *         name: tableName
 *         required: true
 *         schema:
 *           type: string
 *       - in: path
 *         name: columnName
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ConnectionCredentials'
 *     responses:
 *       200:
 *         description: Distinct values
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 values:
 *                   type: array
 *                   items:
 *                     type: string
 */
router.post('/distinct/:tableName/:columnName', async (req, res) => {
    const { tableName, columnName } = req.params;
    const { server, database, username, password } = req.body;

    let pool;
    try {
        pool = await testConnection({ server, database, username, password });

        const result = await pool.request().query(`
      SELECT DISTINCT COALESCE(NULLIF(LTRIM(RTRIM([${columnName}])), ''), '(Empty)') AS [${columnName}]
      FROM [${tableName}]
      ORDER BY [${columnName}]
    `);

        const values = result.recordset.map(row => row[columnName]);

        await closeConnection(pool);
        res.json({ values });
    } catch (error) {
        if (pool) await closeConnection(pool);
        res.status(500).json({ error: error.message });
    }
});

export default router;
