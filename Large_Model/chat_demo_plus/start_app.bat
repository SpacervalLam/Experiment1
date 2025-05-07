@echo off

REM 启动 Ollama
start /b ollama serve

timeout /t 5 > nul

REM 启动本地服务器
start /b node server.js

timeout /t 2 > nul

REM 打开应用页面
start http://localhost:8000

echo All services started.