#!/usr/bin/env python3
"""
Test script for Voice Detection API
Run this to verify the API works before deployment
"""

import requests
import os
from pathlib import Path

API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "your-secret-api-key-change-this-in-production")

def test_health():
    """Test health check endpoint"""
    print("\n🏥 Testing Health Check...")
    try:
        response = requests.get(
            f"{API_URL}/health",
            headers={"X-API-Key": API_KEY}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n📍 Testing Root Endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict(audio_file):
    """Test prediction endpoint"""
    print(f"\n🎙️  Testing Prediction with: {audio_file}")
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{API_URL}/predict",
                headers={"X-API-Key": API_KEY},
                files=files
            )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_invalid_api_key():
    """Test with invalid API key"""
    print("\n🔐 Testing Invalid API Key...")
    try:
        response = requests.get(
            f"{API_URL}/health",
            headers={"X-API-Key": "invalid-key"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 403:
            print("✅ Correctly rejected invalid API key")
            return True
        else:
            print("❌ Should have rejected invalid API key")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("🎵 Voice Detection API - Test Suite")
    print("=" * 60)
    print(f"\nAPI URL: {API_URL}")
    print(f"API Key: {API_KEY[:10]}... (masked)")
    
    results = []
    
    # Run tests
    results.append(("Root Endpoint", test_root()))
    results.append(("Health Check", test_health()))
    results.append(("Invalid API Key", test_invalid_api_key()))
    
    # Find test audio file
    test_audio_paths = [
        "test_audio.wav",
        "sample.wav",
        "audio.wav"
    ]
    
    audio_found = False
    for path in test_audio_paths:
        if Path(path).exists():
            results.append(("Prediction", test_predict(path)))
            audio_found = True
            break
    
    if not audio_found:
        print("\n⚠️  No test audio file found (test_audio.wav, sample.wav, or audio.wav)")
        print("Place a .wav file in the project root to test predictions")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! API is ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
