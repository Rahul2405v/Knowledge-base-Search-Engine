#!/usr/bin/env python3
"""
Interactive test script for file uploads

Usage: python interactive_test.py
"""

import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def upload_file_interactive():
    """Interactive file upload with user input"""
    
    print("📁 Available files in current directory:")
    current_dir = Path(".")
    files = [f for f in current_dir.glob("*") if f.is_file() and f.suffix in ['.txt', '.pdf']]
    
    if not files:
        print("   No .txt or .pdf files found")
        return
    
    for i, file in enumerate(files, 1):
        print(f"   {i}. {file.name}")
    
    try:
        choice = int(input("\nEnter file number to upload (0 to skip): "))
        if choice == 0:
            return
        
        file_path = files[choice - 1]
        
        print(f"\n📤 Uploading {file_path.name}...")
        
        with open(file_path, 'rb') as f:
            files_dict = {'file': (file_path.name, f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/ingest", files=files_dict)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! {result['message']}")
        else:
            print(f"❌ Failed: {response.text}")
            
    except (ValueError, IndexError, FileNotFoundError, requests.RequestException) as e:
        print(f"❌ Error: {e}")

def query_interactive():
    """Interactive querying"""
    
    while True:
        question = input("\n❓ Enter your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            data = {"q": question, "top_k": 3}
            response = requests.post(f"{BASE_URL}/query", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n💡 Answer:")
                print(f"   {result['answer']}")
                
                if result['passages']:
                    print(f"\n📄 Found {len(result['passages'])} relevant passages:")
                    for i, passage in enumerate(result['passages'], 1):
                        print(f"   {i}. {passage['text'][:100]}...")
                        print(f"      Source: {passage['source']}")
                
            else:
                print(f"❌ Query failed: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ Error: {e}")

def main():
    """Main interactive menu"""
    
    # Check server
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server running: {response.json()['message']}")
    except requests.RequestException:
        print("❌ Server not running. Start with: uvicorn app.main:app --reload")
        return
    
    while True:
        print("\n" + "="*50)
        print("🧪 Knowledge Base Test Interface")
        print("="*50)
        print("1. Upload a file")
        print("2. Ask questions")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == '1':
            upload_file_interactive()
        elif choice == '2':
            query_interactive()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()