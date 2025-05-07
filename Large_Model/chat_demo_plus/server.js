const http = require('http');
const fs = require('fs');
const path = require('path');

// 设置服务器端口
const PORT = 8000;

// 创建 HTTP 服务器
const server = http.createServer((req, res) => {
    // 获取请求的文件路径
    let filePath = path.join(__dirname, req.url === '/' ? '/index.html' : req.url);

    // 获取文件扩展名
    const extname = path.extname(filePath);

    // 设置 MIME 类型
    const mimeTypes = {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.wav': 'audio/wav',
        '.mp4': 'video/mp4',
        '.woff': 'application/font-woff',
        '.ttf': 'application/font-ttf',
        '.eot': 'application/vnd.ms-fontobject',
        '.otf': 'application/font-otf',
        '.wasm': 'application/wasm',
        '.txt': 'text/plain',
    };

    const contentType = mimeTypes[extname] || 'application/octet-stream';

    // 读取文件并返回内容
    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                // 文件未找到
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end('<h1>404 Not Found</h1>', 'utf-8');
            } else {
                // 其他服务器错误
                res.writeHead(500);
                res.end(`Server Error: ${error.code}`, 'utf-8');
            }
        } else {
            // 成功返回文件内容
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

// 启动服务器
server.listen(PORT, () => {
    console.log(`服务器已启动，访问地址：http://localhost:${PORT}`);
});