# -*- coding: utf-8 -*-
"""
Test script to verify integrated FLIMageFileReader2 works with both libtiff and tifffile
"""

import os
import sys
import numpy as np

# Test with libtiff mode (default if available)
print("="*80)
print("Testing FLIMageFileReader2 with libtiff mode (if available)")
print("="*80)
from FLIMageFileReader2 import FileReader, USE_TIFFFILE, LIBTIFF_AVAILABLE
print(f"USE_TIFFFILE: {USE_TIFFFILE}")
print(f"LIBTIFF_AVAILABLE: {LIBTIFF_AVAILABLE}")

test_dir = r"C:\Users\WatabeT\Desktop\sample_flim20260110copied"
if os.path.exists(test_dir):
    flim_files = [f for f in os.listdir(test_dir) if f.endswith('.flim')]
    if flim_files:
        file_path = os.path.join(test_dir, flim_files[0])
        print(f"\nTesting file: {os.path.basename(file_path)}")
        try:
            reader = FileReader()
            reader.read_imageFile(file_path, readImage=True)
            print(f"  Successfully read {reader.n_images} images")
            print(f"  Image shape: {np.array(reader.image).shape}")
            print(f"  [OK] Reading successful")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

# Test with tifffile mode (force)
print("\n" + "="*80)
print("Testing FLIMageFileReader2 with tifffile mode (forced)")
print("="*80)

# Force use of tifffile
import FLIMageFileReader2
FLIMageFileReader2.USE_TIFFFILE = True
print(f"USE_TIFFFILE (forced): {FLIMageFileReader2.USE_TIFFFILE}")

# Reload to get updated FileReader
import importlib
importlib.reload(FLIMageFileReader2)
from FLIMageFileReader2 import FileReader

if os.path.exists(test_dir):
    if flim_files:
        file_path = os.path.join(test_dir, flim_files[0])
        print(f"\nTesting file: {os.path.basename(file_path)}")
        try:
            reader = FileReader()
            reader.read_imageFile(file_path, readImage=True)
            print(f"  Successfully read {reader.n_images} images")
            print(f"  Image shape: {np.array(reader.image).shape}")
            print(f"  [OK] Reading successful with tifffile")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

print("\n" + "="*80)
print("Test completed")
print("="*80)
