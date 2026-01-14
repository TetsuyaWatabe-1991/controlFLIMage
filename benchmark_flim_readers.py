# -*- coding: utf-8 -*-
"""
Benchmark script to compare reading speed between FLIMageFileReader2 (libtiff) and FLIMageFileReader3 (tifffile)
"""

import os
import time
import numpy as np
from FLIMageFileReader2 import FileReader as FileReader2
from FLIMageFileReader3 import FileReader as FileReader3

def benchmark_file(file_path, num_runs=3):
    """Benchmark a single file"""
    print(f"\n{'='*80}")
    print(f"Benchmarking file: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    # Benchmark Reader2 (libtiff)
    print("\nBenchmarking FLIMageFileReader2 (libtiff)...")
    times2 = []
    for i in range(num_runs):
        reader2 = FileReader2()
        start_time = time.perf_counter()
        reader2.read_imageFile(file_path, readImage=True)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times2.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f} seconds")
    
    avg_time2 = np.mean(times2)
    std_time2 = np.std(times2)
    print(f"  Average: {avg_time2:.4f} seconds (std: {std_time2:.4f})")
    
    # Benchmark Reader3 (tifffile)
    print("\nBenchmarking FLIMageFileReader3 (tifffile)...")
    times3 = []
    for i in range(num_runs):
        reader3 = FileReader3()
        start_time = time.perf_counter()
        reader3.read_imageFile(file_path, readImage=True)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times3.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f} seconds")
    
    avg_time3 = np.mean(times3)
    std_time3 = np.std(times3)
    print(f"  Average: {avg_time3:.4f} seconds (std: {std_time3:.4f})")
    
    # Compare
    print("\n" + "="*80)
    speedup = avg_time2 / avg_time3
    if speedup > 1:
        print(f"libtiff is {speedup:.2f}x FASTER than tifffile")
    elif speedup < 1:
        print(f"tifffile is {1/speedup:.2f}x FASTER than libtiff")
    else:
        print("Both readers have similar performance")
    
    print(f"libtiff: {avg_time2:.4f}s, tifffile: {avg_time3:.4f}s")
    print("="*80)
    
    return avg_time2, avg_time3, speedup

def main():
    """Main benchmark function"""
    # Test directory
    test_dir = r"C:\Users\WatabeT\Desktop\sample_flim20260110copied"
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Find all .flim files
    flim_files = [f for f in os.listdir(test_dir) if f.endswith('.flim')]
    
    if not flim_files:
        print(f"No .flim files found in {test_dir}")
        return
    
    print(f"Found {len(flim_files)} .flim files")
    print(f"Benchmarking first file: {flim_files[0]}")
    print(f"Number of runs per reader: 3")
    
    # Benchmark first file
    file_path = os.path.join(test_dir, flim_files[0])
    avg_time2, avg_time3, speedup = benchmark_file(file_path, num_runs=3)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if speedup >= 2.0:
        print(f"[RECOMMENDATION] libtiff is {speedup:.2f}x faster (>= 2x)")
        print("  -> Keep using libtiff for better performance")
    elif speedup <= 0.5:
        print(f"[RECOMMENDATION] tifffile is {1/speedup:.2f}x faster")
        print("  -> Consider switching to tifffile")
    else:
        print(f"[RECOMMENDATION] Performance difference is less than 2x (ratio: {speedup:.2f})")
        print("  -> Can use either library, tifffile is more portable")
    print("="*80)

if __name__ == "__main__":
    main()
