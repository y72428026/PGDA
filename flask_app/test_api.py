#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify the Flask API functionality
"""

import requests
import os
import sys

def test_flask_api():
    """Test the Flask API endpoints"""
    
    # Test if server is running
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✓ Flask server is running successfully")
            print("✓ Homepage is accessible")
        else:
            print("✗ Flask server returned status:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("✗ Cannot connect to Flask server:", e)
        print("Please make sure to run 'python3 app.py' first")
        return False
    
    # Test API endpoint with a test image
    test_image_path = '/tmp/test_image.jpg'
    if not os.path.exists(test_image_path):
        print("✗ Test image not found. Creating test image...")
        # Create test image if it doesn't exist
        import cv2
        import numpy as np
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, img)
        print("✓ Test image created")
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post('http://localhost:5000/equalize', files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✓ Image processing API is working correctly")
                print("✓ Histogram equalization completed successfully")
                return True
            else:
                print("✗ API returned error:", data.get('error'))
                return False
        else:
            print("✗ API returned status:", response.status_code)
            return False
            
    except requests.exceptions.RequestException as e:
        print("✗ Error testing API:", e)
        return False

if __name__ == '__main__':
    print("Testing Flask Image Histogram Equalization API...")
    print("=" * 50)
    
    success = test_flask_api()
    
    if success:
        print("=" * 50)
        print("🎉 All tests passed! The Flask API is working correctly.")
        print("You can access the web interface at: http://localhost:5000")
    else:
        print("=" * 50)
        print("❌ Some tests failed. Please check the Flask server.")
        sys.exit(1)