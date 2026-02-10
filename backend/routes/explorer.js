import express from 'express';
import { testConnection, closeConnection } from '../config/database.js';

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

    if (!credentials.port) {
        credentials.port = 1433; // Default SQL Server port
    }

    let pool;
    try {
        pool = await testConnection(credentials);

        let query = `SELECT * FROM [${table}]`;
        const whereClauses = [];

        filters.forEach((filter, index) => {
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

        const result = await pool.request().query(query);

        await closeConnection(pool);
        res.json({
            data: result.recordset,
            count: result.recordset.length
        });
    } catch (error) {
        if (pool) await closeConnection(pool);
        res.status(500).json({ error: error.message });
    }
});

export default router;
