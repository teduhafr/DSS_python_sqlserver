import http from 'http';

function testConnection(dbType) {
    console.log(`Testing with dbType: ${dbType}`);

    const postData = JSON.stringify({
        server: '127.0.0.1',
        database: 'testdb',
        username: 'postgres',
        password: 'password',
        port: 5432,
        dbType: dbType
    });

    const options = {
        hostname: 'localhost',
        port: 5000,
        path: '/api/connection/test',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData)
        }
    };

    const req = http.request(options, (res) => {
        console.log(`STATUS: ${res.statusCode}`);
        res.setEncoding('utf8');
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
            console.log('BODY:', data);
        });
    });

    req.on('error', (e) => {
        console.error(`problem with request: ${e.message}`);
    });

    req.write(postData);
    req.end();
}

testConnection('postgres');
