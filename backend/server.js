import express from 'express';
import cors from 'cors';
import swaggerUi from 'swagger-ui-express';
import { swaggerSpec } from './config/swagger.js';
import connectionRouter from './routes/connection.js';
import metadataRouter from './routes/metadata.js';
import pivotRouter from './routes/pivot.js';
import explorerRouter from './routes/explorer.js';

const app = express();
let PORT = process.env.PORT || 5000;

// Custom port for development
if (process.env.NODE_ENV === 'development' && process.env.DEV_PORT) {
    console.log(`Using custom development port: ${process.env.DEV_PORT}`);
    PORT = process.env.DEV_PORT;
}

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();
});

// Swagger documentation
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
    customCss: '.swagger-ui .topbar { display: none }',
    customSiteTitle: 'DSS API Documentation',
}));

// API routes
app.use('/api/connection', connectionRouter);
app.use('/api/metadata', metadataRouter);
app.use('/api/pivot', pivotRouter);
app.use('/api/explorer', explorerRouter);

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'DSS Application API',
        version: '1.0.0',
        documentation: '/api-docs',
        endpoints: {
            connection: '/api/connection',
            metadata: '/api/metadata',
            pivot: '/api/pivot',
            explorer: '/api/explorer',
        },
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: err.message
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Route not found' });
});

// Start server
app.listen(PORT, () => {
    console.log('═══════════════════════════════════════════════════════');
    console.log(`  DSS Backend Server`);
    console.log(`  Port: ${PORT}`);
    console.log(`  API Documentation: http://localhost:${PORT}/api-docs`);
    console.log('═══════════════════════════════════════════════════════');
});

export default app;
