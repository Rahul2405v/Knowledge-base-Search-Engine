#!/usr/bin/env python3
"""
Test script for the Knowledge Base Search Backend API

This script tests both file upload and query functionality.
Make sure the server is running on http://localhost:8000
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Health check: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: uvicorn app.main:app --reload")
        return False

def test_file_upload(file_path):
    """Test file upload"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"\n📤 Uploading file: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'text/plain')}
        
        try:
            response = requests.post(f"{BASE_URL}/ingest", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload successful!")
                print(f"   Inserted: {result['inserted']}")
                print(f"   Message: {result['message']}")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Upload error: {e}")
            return False

def test_query(question, top_k=3):
    """Test query functionality"""
    print(f"\n❓ Querying: {question}")
    
    data = {
        "q": question,
        "top_k": top_k
    }
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful!")
            print(f"   Answer: {result['answer'][:200]}...")
            print(f"   Found {len(result['passages'])} passages")
            
            for i, passage in enumerate(result['passages'][:2]):  # Show first 2
                print(f"   Passage {i+1}: {passage['text'][:100]}...")
                print(f"   Source: {passage['source']} (Score: {passage['score']:.3f})")
            
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Query error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Knowledge Base Search Backend")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        return
    
    # Test 2: Upload file
    test_file = "test_document.txt"
    if test_file_upload(test_file):
        
        # Test 3: Query after upload
        test_questions = [
            "What is machine learning?",
            "Tell me about deep learning",
            "How does natural language processing work?"
        ]
        
        for question in test_questions:
            test_query(question)
    
    print("\n🎉 Testing complete!")

if __name__ == "__main__":
    main()