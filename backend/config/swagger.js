import swaggerJsdoc from 'swagger-jsdoc';

const options = {
    definition: {
        openapi: '3.0.0',
        info: {
            title: 'DSS Application API',
            version: '1.0.0',
            description: 'RESTful API for Dynamic Decision Support System with Pivot Tables and Data Exploration',
            contact: {
                name: 'API Support',
            },
        },
        servers: [
            {
                url: 'http://localhost:5000',
                description: 'Development server',
            },
        ],
        components: {
            schemas: {
                ConnectionCredentials: {
                    type: 'object',
                    required: ['server', 'database', 'username', 'password'],
                    properties: {
                        server: {
                            type: 'string',
                            example: 'your_server.database.windows.net',
                            description: 'SQL Server hostname or IP',
                        },
                        database: {
                            type: 'string',
                            example: 'your_database',
                            description: 'Database name',
                        },
                        username: {
                            type: 'string',
                            example: 'sa',
                            description: 'Database username',
                        },
                        password: {
                            type: 'string',
                            example: 'password123',
                            description: 'Database password',
                        },
                        port: {
                            type: 'integer',
                            example: 5432,
                            description: 'Database port (default: 5432 for PostgreSQL)',
                        },
                    },
                },
                PivotParams: {
                    type: 'object',
                    required: ['credentials', 'table', 'rows', 'pivotCol', 'value', 'aggFunc'],
                    properties: {
                        credentials: {
                            $ref: '#/components/schemas/ConnectionCredentials',
                        },
                        table: {
                            type: 'string',
                            example: 'Sales',
                            description: 'Table name to pivot',
                        },
                        rows: {
                            type: 'array',
                            items: { type: 'string' },
                            example: ['Region', 'Product'],
                            description: 'Row dimensions',
                        },
                        pivotCol: {
                            type: 'string',
                            example: 'Year',
                            description: 'Column to pivot on',
                        },
                        value: {
                            type: 'string',
                            example: 'Revenue',
                            description: 'Value column to aggregate',
                        },
                        aggFunc: {
                            type: 'string',
                            enum: ['SUM', 'AVG', 'COUNT', 'MAX', 'MIN'],
                            example: 'SUM',
                            description: 'Aggregation function',
                        },
                        showRowTotals: {
                            type: 'boolean',
                            default: false,
                            description: 'Show row totals (horizontal)',
                        },
                        showColTotals: {
                            type: 'boolean',
                            default: false,
                            description: 'Show column totals (vertical)',
                        },
                    },
                },
                DrillDownParams: {
                    type: 'object',
                    required: ['credentials', 'table', 'pivotParams', 'clickedRowData', 'clickedColName'],
                    properties: {
                        credentials: {
                            $ref: '#/components/schemas/ConnectionCredentials',
                        },
                        table: {
                            type: 'string',
                            example: 'Sales',
                        },
                        pivotParams: {
                            type: 'object',
                            description: 'Original pivot parameters',
                        },
                        clickedRowData: {
                            type: 'object',
                            description: 'Data from the clicked row',
                        },
                        clickedColName: {
                            type: 'string',
                            description: 'Name of the clicked column',
                        },
                    },
                },
                FilterParams: {
                    type: 'object',
                    required: ['credentials', 'table'],
                    properties: {
                        credentials: {
                            $ref: '#/components/schemas/ConnectionCredentials',
                        },
                        table: {
                            type: 'string',
                            example: 'Sales',
                        },
                        filters: {
                            type: 'array',
                            items: {
                                type: 'object',
                                properties: {
                                    column: { type: 'string' },
                                    type: {
                                        type: 'string',
                                        enum: ['text_contains', 'numeric_gte', 'numeric_lte', 'numeric_range'],
                                    },
                                    value: {
                                        oneOf: [
                                            { type: 'string' },
                                            { type: 'number' },
                                            { type: 'array', items: { type: 'number' } },
                                        ],
                                    },
                                },
                            },
                        },
                    },
                },
                Error: {
                    type: 'object',
                    properties: {
                        error: {
                            type: 'string',
                            description: 'Error message',
                        },
                    },
                },
            },
        },
    },
    apis: ['./routes/*.js'], // Path to the API routes
};

export const swaggerSpec = swaggerJsdoc(options);
