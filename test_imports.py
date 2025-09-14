#!/usr/bin/env python3
"""
Test script to identify import issues
"""

import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Python path:", sys.path[:3])

try:
    print("Testing basic imports...")
    import os
    print("✓ os imported")
    
    import cv2
    print("✓ cv2 imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import tensorflow as tf
    print("✓ tensorflow imported")
    
    import tensorflow_hub as hub
    print("✓ tensorflow_hub imported")
    
    from flask import Flask
    print("✓ Flask imported")
    
    from flask_jwt_extended import JWTManager
    print("✓ Flask-JWT-Extended imported")
    
    from flask_sqlalchemy import SQLAlchemy
    print("✓ Flask-SQLAlchemy imported")
    
    from flask_migrate import Migrate
    print("✓ Flask-Migrate imported")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
