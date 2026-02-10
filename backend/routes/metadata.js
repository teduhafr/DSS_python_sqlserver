import express from 'express';
import { testConnection, closeConnection } from '../config/database.js';

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
    const { server, database, username, password, port } = req.body;

    if (!port) {
        port = 1433; // Default SQL Server port
    }

    let pool;
    try {
        pool = await testConnection({ server, database, username, password, port });

        // Get sample data to determine columns
        const sampleResult = await pool.request().query(`SELECT TOP 1 * FROM [${tableName}]`);
        const columns = Object.keys(sampleResult.recordset[0] || {});

        // Get column types from schema
        const columnInfoResult = await pool.request().query(`
      SELECT COLUMN_NAME, DATA_TYPE 
      FROM INFORMATION_SCHEMA.COLUMNS 
      WHERE TABLE_NAME = '${tableName}'
    `);

        const searchableColumns = [];
        const numericColumns = [];

        columnInfoResult.recordset.forEach(col => {
            const dataType = col.DATA_TYPE.toLowerCase();

            if (dataType.includes('char') || dataType.includes('text')) {
                searchableColumns.push(col.COLUMN_NAME);
            }

            if (['int', 'float', 'decimal', 'numeric', 'money', 'bigint', 'smallint', 'tinyint', 'real'].some(t => dataType.includes(t))) {
                numericColumns.push(col.COLUMN_NAME);
            }
        });

        await closeConnection(pool);
        res.json({ columns, searchableColumns, numericColumns });
    } catch (error) {
        if (pool) await closeConnection(pool);
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
