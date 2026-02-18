# Testing Manual - Agentic RAG Chatbot
## PRIVATE - Do Not Commit

---

## 🚀 Quick Start for Testing

### 1. Environment Setup
```bash
# Navigate to project
cd C:\Users\thiss\hackathon\agentic-rag-chatbot-SudheerNaraharisetty

# Add API keys to .env
echo "NVIDIA_API_KEY=nvapi-your-key-here" >> .env
echo "GROQ_API_KEY=gsk-your-key-here" >> .env

# Install dependencies (if not done)
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Feature A: RAG + Citations Testing

### Step 1: Upload Documents
1. Open Streamlit app in browser
2. In sidebar, click "Browse files"
3. Upload test documents (PDF, TXT, MD)
4. Wait for "✅ X document(s) processed" message

### Step 2: Test Questions (from EVAL_QUESTIONS.md)

**Test 1: Summarize main contribution**
```
"Summarize the main contribution in 3 bullets."
```
✅ Expect: Grounded summary with citations

**Test 2: Key assumptions/limitations**
```
"What are the key assumptions or limitations?"
```
✅ Expect: Grounded answer with citations

**Test 3: Concrete numeric detail**
```
"Give one concrete numeric/experimental detail and cite it."
```
✅ Expect: Specific claim + citation

**Test 4: Retrieval failure (should refuse)**
```
"What is the CEO's phone number?"
```
✅ Expect: "I don't have enough information..." (no fake citations)

**Test 5: Uncovered topic**
```
"What is the meaning of life?"
```
✅ Expect: "I don't have enough information based on uploaded documents"

---

## 🧠 Feature B: Memory System Testing

### Step 1: Provide User Info
In chat, type:
```
"I am a Project Finance Analyst."
```

### Step 2: Provide Preferences
```
"I prefer weekly summaries on Mondays."
```

### Step 3: Verify Memory
1. Click "🧠 Memory" tab
2. Check USER_MEMORY.md section
3. ✅ Should see entries (not duplicates)

### Step 4: Test Company Memory
Ask about workflows:
```
"How does Asset Management interface with Project Finance?"
```

Check COMPANY_MEMORY.md for learnings.

---

## 🌤️ Feature C: Open-Meteo Natural Language Testing

### Test 1: Basic Weather Query
```
"What's the temperature in New York?"
```
✅ Should geocode "New York" and return current temp

### Test 2: Humidity Comparison
```
"What's the humidity in London last week compared to now?"
```
✅ Should:
1. Geocode "London"
2. Get last week's data
3. Get current data
4. Compare and explain difference

### Test 3: Historical Data
```
"What was the weather in Tokyo yesterday?"
```
✅ Should return yesterday's weather for Tokyo

### Test 4: Time Range
```
"Compare temperature in Paris last month to today"
```
✅ Should handle "last month" → 30 days ago

---

## 🛠️ Testing arXiv Papers

### Where to Download:
1. Go to: https://arxiv.org/
2. Search topics like:
   - "machine learning climate prediction"
   - "natural language processing"
   - "finance risk analysis"
3. Click "Download PDF"

### Recommended Papers (8-10):
Download mix of:
- **3-4 PDFs** (ML/Climate/Finance)
- **2-3 TXTs** (Convert PDF to text or find plain text papers)
- **2-3 MDs** (Markdown documentation or converted papers)

### Where to Place:
```
sample_docs/
├── paper1_ml_climate.pdf
├── paper2_nlp.pdf
├── paper3_finance.txt
├── paper4_weather.md
└── ... (8-10 total files)
```

### Testing Different Formats:

**PDF Test:**
```bash
# Upload any .pdf file
# Ask: "What is the main methodology?"
# Check if citations show page numbers
```

**TXT Test:**
```bash
# Upload any .txt file
# Ask: "Summarize the conclusion"
```

**MD Test:**
```bash
# Upload any .md file
# Ask: "What are the key findings?"
```

---

## 🧪 Running Sanity Check

### Command:
```bash
make sanity
```

### Expected Output:
- Should create: `artifacts/sanity_output.json`
- Should validate format
- Should show: "OK: sanity check passed"

### Manual Validation:
```bash
# Check file exists
ls artifacts/sanity_output.json

# View content
cat artifacts/sanity_output.json
```

Should contain:
- `implemented_features`: ["A", "B", "C"]
- `qa`: Array of Q&A with citations
- `demo`: Memory writes and tool calls

---

## 🐛 Common Issues & Fixes

### Issue 1: "Module not found"
```bash
# Fix: Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Issue 2: "API key error"
```bash
# Fix: Check .env file
cat .env
# Should show: NVIDIA_API_KEY=nvapi-...
```

### Issue 3: "No documents indexed"
```bash
# Fix: Check sample_docs folder
ls sample_docs/
# Upload files via UI
```

### Issue 4: "Memory not writing"
```bash
# Fix: Check file permissions
ls -la USER_MEMORY.md COMPANY_MEMORY.md
# Should be writable
```

---

## 📹 Video Recording Checklist

Before recording walkthrough:
- [ ] All 3 features working
- [ ] Sample documents uploaded
- [ ] API keys working
- [ ] Tested all EVAL_QUESTIONS
- [ ] Memory files showing entries
- [ ] Open-Meteo working with natural language

**Video Structure (5-10 min):**
1. **Intro** (30 sec): What you built
2. **Feature A** (2 min): Upload PDF → Ask questions → Show citations
3. **Feature B** (1 min): Tell bot preferences → Show memory files
4. **Feature C** (2 min): Natural language weather query → Show analysis
5. **Tradeoffs** (1 min): Design decisions
6. **Improvements** (1 min): What you'd add with more time

---

## 🔒 Security Checklist

Before submission:
- [ ] `.env` is in `.gitignore` (YES)
- [ ] No API keys committed (check with: `git log --all --full-history -- .env`)
- [ ] USER_MEMORY.md and COMPANY_MEMORY.md are writable
- [ ] Sanity check passes: `make sanity`

---

## 🎯 Final Verification Steps

1. **Test complete flow:**
   ```bash
   # Fresh start
   rm -rf data/ artifacts/
   streamlit run app.py
   ```

2. **Upload documents**

3. **Ask all EVAL_QUESTIONS**

4. **Check memory files:**
   ```bash
   cat USER_MEMORY.md
   cat COMPANY_MEMORY.md
   ```

5. **Run sanity check:**
   ```bash
   make sanity
   ```

6. **Verify JSON output:**
   ```bash
   cat artifacts/sanity_output.json
   ```

---

## 📞 Emergency Contacts

If something breaks:
1. Check this manual first
2. Check `whatsleft.md` for pending tasks
3. Review error messages in terminal
4. Check GitHub issues (if any)

---

**Last Updated:** 2024-02-17
**For:** Agentic RAG Chatbot Hackathon
**Repository:** https://github.com/Community-Dreams-Foundation/agentic-rag-chatbot-SudheerNaraharisetty