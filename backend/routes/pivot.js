import express from 'express';
import fs from 'fs';
import { testConnection, closeConnection, testPostgresConnection, executePostgresQuery, closePostgresConnection as closePostgres } from '../config/database.js';

const router = express.Router();

/**
 * @swagger
 * /api/pivot/generate:
 *   post:
 *     summary: Generate pivot table
 *     description: Creates a dynamic pivot table based on specified parameters
 *     tags: [Pivot]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/PivotParams'
 *     responses:
 *       200:
 *         description: Pivot table data and SQL query
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                 query:
 *                   type: string
 *                 pivotValues:
 *                   type: array
 *                   items:
 *                     type: string
 *       400:
 *         description: Invalid parameters
 *       500:
 *         description: Failed to generate pivot
 */
router.post('/generate', async (req, res) => {
    const { credentials, table, rows, pivotCol, value, aggFunc, showRowTotals, showColTotals } = req.body;

    if (!table || !rows || !pivotCol || !value || !aggFunc) {
        return res.status(400).json({ error: 'Missing required pivot parameters' });
    }

    const isNumericAgg = ['SUM', 'AVG'].includes(aggFunc);
    const dbType = credentials?.dbType || 'mssql'; // Get dbType from credentials

    let dbClient;
    try {
        if (dbType === 'postgres') {
            const pgPort = parseInt(credentials.port) || 5432;

            try {
                fs.appendFileSync('pivot_debug.log', `Connecting to Postgres: ${JSON.stringify({ ...credentials, password: '***' })}\n`);
                dbClient = await testPostgresConnection({
                    host: credentials.server,
                    database: credentials.database,
                    username: credentials.username,
                    password: credentials.password,
                    port: pgPort
                });
                fs.appendFileSync('pivot_debug.log', 'Connected to Postgres\n');

                // Get distinct values for pivot column
                const distinctQuery = `
                  SELECT DISTINCT COALESCE(NULLIF(TRIM("${pivotCol}"), ''), '(Empty)') AS "${pivotCol}"
                  FROM "${table}"
                  ORDER BY "${pivotCol}";
                `;
                fs.appendFileSync('pivot_debug.log', `Executing distinct query: ${distinctQuery}\n`);

                const distinctResult = await executePostgresQuery(dbClient, distinctQuery);
                fs.appendFileSync('pivot_debug.log', `Distinct query result: ${distinctResult.rows.length}\n`);
                const pivotValues = distinctResult.rows.map(row => row[pivotCol]).filter(v => v);

                if (pivotValues.length === 0) {
                    await closePostgres(dbClient);
                    return res.status(400).json({ error: `No distinct values found for column '${pivotCol}'` });
                }

                // Build dynamic pivot query
                const rowColsStr = rows.map(r => `"${r}"`).join(', ');

                const aggExpressions = pivotValues.map(v => {
                    const safeValue = v === '(Empty)' ? '' : v.replace(/'/g, "''");
                    return `${aggFunc}(CASE WHEN "${pivotCol}" = '${safeValue}' THEN "${value}" END) AS "${v}"`;
                });

                if (showRowTotals) {
                    aggExpressions.push(`${aggFunc}("${value}") AS "Total"`);
                }

                const aggExpressionsStr = aggExpressions.join(',\n       ');

                let selectCols = rowColsStr;
                let groupingLogic = rows.length > 0 ? `GROUP BY ${rowColsStr}` : '';

                if (showColTotals && rows.length > 0) {
                    groupingLogic = `GROUP BY GROUPING SETS ((${rowColsStr}), ())`;
                    const coalesceExpressions = rows.map((r, idx) => {
                        if (idx === 0) {
                            return `COALESCE(CAST("${r}" AS VARCHAR), 'Total') AS "${r}"`;
                        }
                        return `COALESCE(CAST("${r}" AS VARCHAR), '') AS "${r}"`;
                    });
                    selectCols = coalesceExpressions.join(', ');
                }

                const pivotQuery = `
                  SELECT
                    ${selectCols}${selectCols && aggExpressionsStr ? ',' : ''}
                    ${aggExpressionsStr}
                  FROM "${table}"
                  ${groupingLogic}
                `;
                fs.appendFileSync('pivot_debug.log', `Executing pivot query: ${pivotQuery}\n`);

                const result = await executePostgresQuery(dbClient, pivotQuery);
                fs.appendFileSync('pivot_debug.log', `Pivot query result: ${result.rows.length}\n`);

                await closePostgres(dbClient);
                res.json({
                    data: result.rows,
                    query: pivotQuery,
                    pivotValues
                });
            } catch (err) {
                fs.appendFileSync('pivot_debug.log', `Error in Postgres block: ${err.message}\n${err.stack}\n`);
                throw err;
            }
        } else {
            dbClient = await testConnection(credentials);

            // Get distinct values for pivot column
            const distinctQuery = `
      SELECT DISTINCT COALESCE(NULLIF(LTRIM(RTRIM([${pivotCol}])), ''), '(Empty)') AS [${pivotCol}]
      FROM [${table}]
      ORDER BY [${pivotCol}]
    `;

            const distinctResult = await dbClient.request().query(distinctQuery);
            const pivotValues = distinctResult.recordset.map(row => row[pivotCol]).filter(v => v);

            if (pivotValues.length === 0) {
                await closeConnection(dbClient);
                return res.status(400).json({ error: `No distinct values found for column '${pivotCol}'` });
            }

            // Build dynamic pivot query
            const rowColsStr = rows.map(r => `[${r}]`).join(', ');

            const aggExpressions = pivotValues.map(v => {
                const safeValue = v === '(Empty)' ? '' : v.replace(/'/g, "''");
                return `${aggFunc}(CASE WHEN [${pivotCol}] = '${safeValue}' THEN [${value}] END) AS [${v}]`;
            });

            if (showRowTotals) {
                aggExpressions.push(`${aggFunc}([${value}]) AS [Total]`);
            }

            const aggExpressionsStr = aggExpressions.join(',\n       ');

            let selectCols = rowColsStr;
            let groupingLogic = rows.length > 0 ? `GROUP BY ${rowColsStr}` : '';

            if (showColTotals && rows.length > 0) {
                groupingLogic = `GROUP BY GROUPING SETS ((${rowColsStr}), ())`;
                const coalesceExpressions = rows.map((r, idx) => {
                    if (idx === 0) {
                        return `COALESCE(CAST([${r}] AS VARCHAR(MAX)), 'Total') AS [${r}]`;
                    }
                    return `COALESCE(CAST([${r}] AS VARCHAR(MAX)), '') AS [${r}]`;
                });
                selectCols = coalesceExpressions.join(', ');
            }

            const pivotQuery = `
      SELECT
        ${selectCols}${selectCols && aggExpressionsStr ? ',' : ''}
        ${aggExpressionsStr}
      FROM [${table}]
      ${groupingLogic}
    `;

            const result = await dbClient.request().query(pivotQuery);

            await closeConnection(dbClient);
            res.json({
                data: result.recordset,
                query: pivotQuery,
                pivotValues
            });
        }
    } catch (error) {
        return res.status(500).json({ error: error.message });
    } finally {
        if (dbType === 'postgres' && dbClient) {
            await closePostgres(dbClient);
        } else if (dbClient) {
            await closeConnection(dbClient);
        }
    }
});

/**
 * @swagger
 * /api/pivot/drilldown:
 *   post:
 *     summary: Get drill-down data
 *     description: Fetches raw data for a specific pivot cell
 *     tags: [Pivot]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/DrillDownParams'
 *     responses:
 *       200:
 *         description: Drill-down data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                 filterInfo:
 *                   type: object
 *       500:
 *         description: Failed to fetch drill-down data
 */
router.post('/drilldown', async (req, res) => {
    const { credentials, table, pivotParams, clickedRowData, clickedColName } = req.body;

    let client;
    try {
        if (credentials.dbType === 'postgres') {
            const pgPort = parseInt(credentials.port) || 5432;
            client = await testPostgresConnection({
                host: credentials.server,
                database: credentials.database,
                username: credentials.username,
                password: credentials.password,
                port: pgPort
            });

            const { rows, pivotCol, showRowTotals, showColTotals } = pivotParams;
            const isTotalRow = showColTotals && rows.length > 0 && clickedRowData[rows[0]] === 'Total';
            const isTotalCol = showRowTotals && clickedColName === 'Total';

            const whereClauses = [];

            if (!isTotalRow) {
                rows.forEach(col => {
                    if (clickedRowData[col] !== null && clickedRowData[col] !== undefined) {
                        const value = clickedRowData[col].toString().replace(/'/g, "''");
                        whereClauses.push(`"${col}" = '${value}'`);
                    }
                });
            }

            if (!isTotalCol) {
                const colValue = clickedColName === '(Empty)' ? '' : clickedColName.replace(/'/g, "''");
                whereClauses.push(`COALESCE(NULLIF(TRIM("${pivotCol}"), ''), '(Empty)') = '${clickedColName}'`);
            }

            const whereStr = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';
            // Limit for safety
            const query = `SELECT * FROM "${table}" ${whereStr} LIMIT 1000`;

            const result = await executePostgresQuery(client, query);

            const filterInfo = {
                ...Object.fromEntries(rows.map(col => [col, clickedRowData[col]])),
                [pivotCol]: clickedColName
            };

            await closePostgres(client);
            res.json({
                data: result.rows,
                filterInfo
            });

        } else {
            // Default to MSSQL
            pool = await testConnection(credentials);

            const { rows, pivotCol, showRowTotals, showColTotals } = pivotParams;

            const isTotalRow = showColTotals && rows.length > 0 && clickedRowData[rows[0]] === 'Total';
            const isTotalCol = showRowTotals && clickedColName === 'Total';

            const whereClauses = [];

            if (!isTotalRow) {
                rows.forEach(col => {
                    if (clickedRowData[col] !== null && clickedRowData[col] !== undefined) {
                        const value = clickedRowData[col].toString().replace(/'/g, "''");
                        whereClauses.push(`[${col}] = '${value}'`);
                    }
                });
            }

            if (!isTotalCol) {
                const colValue = clickedColName === '(Empty)' ? '' : clickedColName.replace(/'/g, "''");
                whereClauses.push(`COALESCE(NULLIF(LTRIM(RTRIM([${pivotCol}])), ''), '(Empty)') = '${clickedColName}'`);
            }

            const whereStr = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';
            const query = `SELECT TOP 1000 * FROM [${table}] ${whereStr}`;

            const result = await pool.request().query(query);

            const filterInfo = {
                ...Object.fromEntries(rows.map(col => [col, clickedRowData[col]])),
                [pivotCol]: clickedColName
            };

            await closeConnection(pool);
            res.json({
                data: result.recordset,
                filterInfo
            });
        }
    } catch (error) {
        if (client) await closePostgres(client);
        if (pool) await closeConnection(pool);
        res.status(500).json({ error: error.message });
    }
});

export default router;
