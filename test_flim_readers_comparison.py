# -*- coding: utf-8 -*-
"""
Test script to compare FLIMageFileReader2 (libtiff) and FLIMageFileReader3 (tifffile)
"""

import os
import sys
import numpy as np
from FLIMageFileReader2 import FileReader as FileReader2
from FLIMageFileReader3 import FileReader as FileReader3

def compare_attributes(obj1, obj2, path="", differences=None):
    """Recursively compare attributes of two objects"""
    if differences is None:
        differences = []
    
    # Get all attributes
    attrs1 = set(dir(obj1))
    attrs2 = set(dir(obj2))
    
    # Compare common attributes
    common_attrs = attrs1 & attrs2
    
    for attr in common_attrs:
        # Skip private attributes and methods
        if attr.startswith('_') or callable(getattr(obj1, attr, None)):
            continue
        
        try:
            val1 = getattr(obj1, attr)
            val2 = getattr(obj2, attr)
            
            current_path = f"{path}.{attr}" if path else attr
            
            # Handle numpy arrays
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    differences.append(f"{current_path}: shapes differ or values differ")
                    differences.append(f"  Reader2: shape={val1.shape}, dtype={val1.dtype}, sample={val1.flat[:min(5, val1.size)]}")
                    differences.append(f"  Reader3: shape={val2.shape}, dtype={val2.dtype}, sample={val2.flat[:min(5, val2.size)]}")
            # Handle lists
            elif isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    differences.append(f"{current_path}: list lengths differ ({len(val1)} vs {len(val2)})")
                else:
                    for i, (v1, v2) in enumerate(zip(val1, val2)):
                        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                            if not np.array_equal(v1, v2):
                                differences.append(f"{current_path}[{i}]: arrays differ")
                        elif isinstance(v1, float) and isinstance(v2, float) and (np.isnan(v1) and np.isnan(v2)):
                            # Both are NaN, consider them equal
                            pass
                        elif v1 != v2:
                            differences.append(f"{current_path}[{i}]: values differ ({v1} vs {v2})")
            # Handle nested objects
            elif hasattr(val1, '__dict__') and hasattr(val2, '__dict__'):
                compare_attributes(val1, val2, current_path, differences)
            # Handle simple values
            else:
                # Handle NaN comparison
                if isinstance(val1, float) and isinstance(val2, float) and (np.isnan(val1) and np.isnan(val2)):
                    # Both are NaN, consider them equal
                    pass
                elif val1 != val2:
                    differences.append(f"{current_path}: {val1} != {val2}")
        except Exception as e:
            differences.append(f"{current_path}: Error comparing - {e}")
    
    return differences

def test_file(file_path):
    """Test a single file"""
    print(f"\n{'='*80}")
    print(f"Testing file: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    try:
        # Read with Reader2 (libtiff)
        print("\nReading with FLIMageFileReader2 (libtiff)...")
        reader2 = FileReader2()
        reader2.read_imageFile(file_path, readImage=True)
        print(f"  Successfully read {reader2.n_images} images")
        
        # Read with Reader3 (tifffile)
        print("Reading with FLIMageFileReader3 (tifffile)...")
        reader3 = FileReader3()
        reader3.read_imageFile(file_path, readImage=True)
        print(f"  Successfully read {reader3.n_images} images")
        
        # Compare basic attributes
        print("\nComparing basic attributes...")
        basic_attrs = ['n_images', 'flim', 'nChannels', 'width', 'height', 
                       'nFastZSlices', 'FastZStack', 'ZStack', 'ImageFormat']
        basic_diffs = []
        for attr in basic_attrs:
            val2 = getattr(reader2, attr, None)
            val3 = getattr(reader3, attr, None)
            if val2 != val3:
                basic_diffs.append(f"  {attr}: Reader2={val2}, Reader3={val3}")
        
        if basic_diffs:
            print("  Differences found in basic attributes:")
            for diff in basic_diffs:
                print(diff)
        else:
            print("  [OK] Basic attributes match")
        
        # Compare State parameters
        print("\nComparing State parameters...")
        state_diffs = compare_attributes(reader2.State, reader3.State, "State")
        if state_diffs:
            print(f"  Found {len(state_diffs)} differences in State parameters:")
            for diff in state_diffs[:20]:  # Show first 20 differences
                print(f"  {diff}")
            if len(state_diffs) > 20:
                print(f"  ... and {len(state_diffs) - 20} more differences")
        else:
            print("  [OK] State parameters match")
        
        # Compare acqTime
        print("\nComparing acquisition times...")
        if reader2.acqTime != reader3.acqTime:
            print(f"  acqTime differs:")
            print(f"    Reader2: {reader2.acqTime}")
            print(f"    Reader3: {reader3.acqTime}")
        else:
            print(f"  [OK] Acquisition times match ({len(reader2.acqTime)} entries)")
        
        # Compare images
        print("\nComparing image data...")
        if len(reader2.image) != len(reader3.image):
            print(f"  Number of images differs: Reader2={len(reader2.image)}, Reader3={len(reader3.image)}")
        else:
            image_diffs = []
            for page_idx in range(len(reader2.image)):
                img2 = reader2.image[page_idx]
                img3 = reader3.image[page_idx]
                
                if isinstance(img2, list) and isinstance(img3, list):
                    if len(img2) != len(img3):
                        image_diffs.append(f"  Page {page_idx}: number of fastZ slices differs ({len(img2)} vs {len(img3)})")
                    else:
                        for fastz_idx in range(len(img2)):
                            img2_fastz = img2[fastz_idx]
                            img3_fastz = img3[fastz_idx]
                            
                            if isinstance(img2_fastz, list) and isinstance(img3_fastz, list):
                                if len(img2_fastz) != len(img3_fastz):
                                    image_diffs.append(f"  Page {page_idx}, FastZ {fastz_idx}: number of channels differs ({len(img2_fastz)} vs {len(img3_fastz)})")
                                else:
                                    for ch_idx in range(len(img2_fastz)):
                                        arr2 = img2_fastz[ch_idx]
                                        arr3 = img3_fastz[ch_idx]
                                        
                                        if isinstance(arr2, np.ndarray) and isinstance(arr3, np.ndarray):
                                            if arr2.shape != arr3.shape:
                                                image_diffs.append(f"  Page {page_idx}, FastZ {fastz_idx}, Channel {ch_idx}: shapes differ ({arr2.shape} vs {arr3.shape})")
                                            else:
                                                # Check if arrays are exactly equal
                                                if not np.array_equal(arr2, arr3):
                                                    # Calculate detailed statistics
                                                    diff_mask = arr2 != arr3
                                                    num_different = np.sum(diff_mask)
                                                    total_pixels = arr2.size
                                                    max_diff = np.max(np.abs(arr2.astype(np.float64) - arr3.astype(np.float64)))
                                                    min_diff = np.min(np.abs(arr2.astype(np.float64) - arr3.astype(np.float64)))
                                                    mean_diff = np.mean(np.abs(arr2.astype(np.float64) - arr3.astype(np.float64)))
                                                    
                                                    # Find first difference location
                                                    diff_indices = np.where(diff_mask)
                                                    if len(diff_indices[0]) > 0:
                                                        first_diff_idx = tuple(idx[0] for idx in diff_indices)
                                                        val2_at_diff = arr2[first_diff_idx]
                                                        val3_at_diff = arr3[first_diff_idx]
                                                        
                                                        image_diffs.append(f"  Page {page_idx}, FastZ {fastz_idx}, Channel {ch_idx}: pixel values differ")
                                                        image_diffs.append(f"    Total pixels: {total_pixels}, Different pixels: {num_different} ({100.0*num_different/total_pixels:.4f}%)")
                                                        image_diffs.append(f"    Max diff: {max_diff}, Min diff: {min_diff}, Mean diff: {mean_diff:.6f}")
                                                        image_diffs.append(f"    First difference at {first_diff_idx}: Reader2={val2_at_diff}, Reader3={val3_at_diff}")
                                                    else:
                                                        image_diffs.append(f"  Page {page_idx}, FastZ {fastz_idx}, Channel {ch_idx}: arrays differ but no differences found (dtype issue?)")
                                                # Also check dtype
                                                elif arr2.dtype != arr3.dtype:
                                                    image_diffs.append(f"  Page {page_idx}, FastZ {fastz_idx}, Channel {ch_idx}: dtypes differ ({arr2.dtype} vs {arr3.dtype}) but values match")
                                        elif arr2 != arr3:
                                            image_diffs.append(f"  Page {page_idx}, FastZ {fastz_idx}, Channel {ch_idx}: types differ")
            
            if image_diffs:
                print(f"  Found differences in image data:")
                for diff in image_diffs:  # Show all differences for detailed analysis
                    print(diff)
            else:
                print("  [OK] Image data matches - all pixel values are identical")
        
        # Summary
        print("\n" + "="*80)
        total_diffs = len(basic_diffs) + len(state_diffs) + (0 if reader2.acqTime == reader3.acqTime else 1) + len(image_diffs) if 'image_diffs' in locals() else 0
        if total_diffs == 0:
            print("[PASS] ALL TESTS PASSED - Readers produce identical results!")
        else:
            print(f"[WARN] Found {total_diffs} differences between readers")
        print("="*80)
        
        return total_diffs == 0
        
    except Exception as e:
        print(f"\n[ERROR] Error testing file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
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
    print(f"Testing first file: {flim_files[0]}")
    
    # Test first file
    file_path = os.path.join(test_dir, flim_files[0])
    success = test_file(file_path)
    
    # Ask if user wants to test more files
    if len(flim_files) > 1:
        print(f"\n{len(flim_files) - 1} more files available. Test all? (y/n): ", end='')
        # For automated testing, test all files
        test_all = True  # Set to False if you want interactive mode
        
        if test_all:
            print("y")
            for flim_file in flim_files[1:]:
                file_path = os.path.join(test_dir, flim_file)
                test_file(file_path)

if __name__ == "__main__":
    main()
