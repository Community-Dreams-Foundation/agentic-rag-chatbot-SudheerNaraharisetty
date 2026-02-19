import requests
import json
import sys

def test_chat():
    url = "http://localhost:8000/api/chat"
    payload = {
        "message": "Who are you?",
        "history": [],
        "model": "openrouter"
    }
    
    print(f"Connecting to {url}...")
    try:
        with requests.post(url, json=payload, stream=True) as r:
            print(f"Status Code: {r.status_code}")
            if r.status_code != 200:
                print(f"Error: {r.text}")
                return

            print("--- Stream Start ---")
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    print(f"CHUNK: {chunk.decode('utf-8', errors='replace')}")
            print("--- Stream End ---")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_chat()
