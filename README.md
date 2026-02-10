# DSS Application - Frontend + Backend Split

## Overview

This project splits the original Streamlit DSS application into a modern architecture:
- **Backend**: Node.js/Express REST API with SQL Server connectivity
- **Frontend**: React application with Tailwind CSS
- **Documentation**: Interactive Swagger/OpenAPI documentation

## Features

✅ Dynamic pivot table generation with T-SQL  
✅ Drill-down functionality for detailed data exploration  
✅ Interactive data filtering and querying  
✅ AG Grid for high-performance data display  
✅ Swagger documentation for all API endpoints  
✅ Modern dark-themed UI with Tailwind CSS  

## Prerequisites

- Node.js 18+ and npm
- SQL Server instance with accessible database
- ODBC Driver 17 for SQL Server (for backend)

## Installation

### Backend Setup

```bash
cd backend
npm install
```

Create a `.env` file (optional):
```bash
cp .env.example .env
# Edit .env with your default database credentials (optional)
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Start Backend Server

```bash
cd backend
npm start
```

The backend API will run on **http://localhost:5000**  
Swagger documentation available at **http://localhost:5000/api-docs**

### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

The frontend will run on **http://localhost:5173**

## API Documentation

Access the interactive Swagger UI at:
```
http://localhost:5000/api-docs
```

### Available Endpoints

#### Connection
- `POST /api/connection/test` - Test database connection
- `POST /api/connection/tables` - Get list of tables

#### Metadata
- `POST /api/metadata/columns/:tableName` - Get column metadata
- `POST /api/metadata/distinct/:tableName/:columnName` - Get distinct values

#### Pivot
- `POST /api/pivot/generate` - Generate dynamic pivot table
- `POST /api/pivot/drilldown` - Get drill-down data

#### Data Explorer
- `POST /api/explorer/query` - Query and filter data

## Architecture

```
dss_python_sqlserver/
├── backend/
│   ├── config/
│   │   ├── database.js      # SQL Server connection
│   │   └── swagger.js       # Swagger configuration
│   ├── routes/
│   │   ├── connection.js    # Connection endpoints
│   │   ├── metadata.js      # Metadata endpoints
│   │   ├── pivot.js         # Pivot endpoints
│   │   └── explorer.js      # Explorer endpoints
│   ├── server.js            # Main Express app
│   └── package.json
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DatabaseConnection.jsx
│   │   │   ├── PivotControls.jsx
│   │   │   ├── PivotGrid.jsx
│   │   │   ├── DrillDown.jsx
│   │   │   └── DataExplorer.jsx
│   │   ├── services/
│   │   │   └── api.js       # API service layer
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
└── dss.py                   # Original Streamlit app (preserved)
```

## Usage

1. **Start both servers** (backend and frontend)
2. **Open frontend** at http://localhost:5173
3. **Connect to database** using the sidebar form
4. **Generate pivot tables** or **explore data** using the tabs
5. **View API documentation** at http://localhost:5000/api-docs

## Original Application

The original Streamlit application (`dss.py`) is preserved in the root directory and can still be run separately:

```bash
streamlit run dss.py
```

## Technologies Used

### Backend
- Express.js - Web framework
- mssql - SQL Server driver
- swagger-jsdoc - Swagger generation
- swagger-ui-express - Swagger UI
- cors - CORS middleware

### Frontend
- React - UI library
- Vite - Build tool
- Tailwind CSS - Styling
- AG Grid - Data grid
- Axios - HTTP client
- Plotly.js - Charts (ready for future enhancements)

## Development

### Backend Development
```bash
cd backend
npm run dev  # Auto-reload on file changes
```

### Frontend Development
```bash
cd frontend
npm run dev  # Hot module replacement
```

### Build for Production
```bash
cd frontend
npm run build
```

## License

ISC
