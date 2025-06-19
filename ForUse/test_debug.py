# Test script to debug the TypeError issue
import sys
import os
sys.path.append("../")

# Test 1: Import and test get_flimfile_list
print("=== Test 1: Import and test get_flimfile_list ===")
try:
    from FLIMageAlignment import get_flimfile_list
    print("✓ Successfully imported get_flimfile_list")
    print(f"get_flimfile_list function: {get_flimfile_list}")
    print(f"get_flimfile_list type: {type(get_flimfile_list)}")
except Exception as e:
    print(f"✗ Failed to import get_flimfile_list: {e}")

# Test 2: Test with a sample file path
print("\n=== Test 2: Test with sample file path ===")
test_file = r"G:\ImagingData\Tetsuya\20250618\titration_dend10_11um_001.flim"
print(f"Test file: {test_file}")
print(f"File exists: {os.path.exists(test_file)}")

try:
    result = get_flimfile_list(test_file)
    print(f"✓ get_flimfile_list returned: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result) if isinstance(result, list) else 'N/A'}")
except Exception as e:
    print(f"✗ get_flimfile_list failed: {e}")

# Test 3: Test the process_and_plot_flim_images function
print("\n=== Test 3: Test process_and_plot_flim_images function ===")
try:
    # Import the function
    from GCaMP_unc_pow_dur_titration_20250618 import process_and_plot_flim_images
    print("✓ Successfully imported process_and_plot_flim_images")
    
    # Test with dummy data
    dummy_filelist = ["dummy_file1.flim", "dummy_file2.flim"]
    dummy_slope = 0.1
    dummy_intercept = 0.05
    
    print("Testing function call...")
    result = process_and_plot_flim_images(dummy_filelist, dummy_slope, dummy_intercept)
    print(f"✓ Function call successful, returned: {type(result)}")
    
except Exception as e:
    print(f"✗ process_and_plot_flim_images test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test completed ===") 