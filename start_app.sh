#!/bin/bash
echo "Starting Agentic RAG System..."

# Start Backend in background
echo "Starting backend on port 8000..."
python -m uvicorn src.api.server:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "Backend ready!"
        break
    fi
    sleep 1
done

# Start Frontend
echo "Starting frontend on port 3000..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "==================================="
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo "==================================="
echo ""
echo "Press Ctrl+C to stop both servers."

# Trap SIGINT to kill both
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
