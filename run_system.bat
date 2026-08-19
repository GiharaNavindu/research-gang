@echo off
echo ======================================================================
echo           AETHELGARD: AGENTIC RESEARCH ENGINE STARTUP SCRIPT
echo ======================================================================
echo.

:: Start FastAPI Backend
echo [1/2] Launching FastAPI Backend Server on port 8000...
start cmd /k "title Aethelgard Backend (FastAPI) && echo Starting FastAPI Backend Server on http://127.0.0.1:8000... && cd /d %~dp0 && python server.py"

:: Start React Frontend
echo [2/2] Launching Vite React TypeScript Frontend on port 5173...
start cmd /k "title Aethelgard Frontend (Vite) && echo Starting React Dev Server on http://127.0.0.1:5173... && cd /d %~dp0\frontend && npm run dev"

echo.
echo ======================================================================
echo Success! Both servers are starting up in separate terminal windows.
echo - Backend API:   http://127.0.0.1:8000
echo - Web Frontend:  http://127.0.0.1:5173
echo.
echo Leave this script and the spawned windows open to continue testing.
echo ======================================================================
pause
