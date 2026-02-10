import fetch from 'node-fetch';

async function testConnection(dbType) {
    console.log(`Testing with dbType: ${dbType}`);
    try {
        const response = await fetch('http://localhost:5000/api/connection/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                server: '127.0.0.1', // Assuming local postgres for test, or just to trigger the driver
                database: 'testdb',
                username: 'postgres',
                password: 'password', // Dummy password
                port: 5432,
                dbType: dbType
            })
        });

        const data = await response.json();
        console.log(`Response status: ${response.status}`);
        console.log('Response body:', data);
    } catch (error) {
        console.error('Fetch error:', error.message);
    }
}

async function run() {
    await testConnection('postgres');
}

run();
