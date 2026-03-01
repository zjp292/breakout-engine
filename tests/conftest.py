"""
Pytest configuration: add project root to sys.path so all test files
can import engine.py, config.py, models.py, etc. directly.
"""
import sys
import os

# Insert project root (one level above this tests/ directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
