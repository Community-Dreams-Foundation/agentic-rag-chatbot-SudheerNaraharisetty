Write-Host "Starting Agentic RAG System..." -ForegroundColor Green

# Start Backend
Write-Host "Starting Backend on port 8000..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\Users\thiss\hackathon\agentic-rag-chatbot-SudheerNaraharisetty; uvicorn src.api.server:app --reload --port 8000"

# Start Frontend
Write-Host "Starting Frontend on port 3000..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\Users\thiss\hackathon\agentic-rag-chatbot-SudheerNaraharisetty\frontend; npm run dev"

Write-Host "Both servers starting. Access the UI at http://localhost:3000" -ForegroundColor Cyan
