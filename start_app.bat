@echo off
echo Starting Agentic RAG System...

:: Start Backend in new window
start "Backend API" cmd /k "uvicorn src.api.server:app --reload --port 8000"

:: Start Frontend in new window
cd frontend
start "Frontend UI" cmd /k "npm run dev"

echo Both servers launching...
echo Backend: http://localhost:8000/docs
echo Frontend: http://localhost:3000
pause
