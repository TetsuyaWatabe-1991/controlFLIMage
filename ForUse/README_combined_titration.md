# GCaMP Uncaging Titration - Combined System

This system combines the functionality of the original `GCaMP_unc_pow_dur_titration_20250618.py` and `GCaMP_unc_fivepulse_titration_20250612.py` scripts into a modular, configurable system.

## Files Overview

### 1. `flim_analysis_utils.py`
**Utility module containing common functions:**
- `calculate_laser_power_settings()`: Converts mW values to power percentages
- `process_flim_image()`: Processes individual FLIM image files
- `plot_single_image()`: Plots individual F/F0 images
- `plot_multiple_images()`: Plots multiple images in a grid with tdTomato reference
- `process_and_plot_flim_images()`: Main function for processing and plotting

### 2. `titration_config.py`
**Configuration file with all experiment parameters:**
- File paths and directories
- Laser power settings
- Setting file paths
- Plotting parameters
- Acquisition timing
- Power calibration settings

### 3. `GCaMP_unc_combined_titration.py`
**Main script that runs the combined experiment:**
- Loads power calibration from JSON or gets manual user input
- Runs titration experiments for multiple setting paths
- Creates separate plots for each setting path
- Supports both single and multiple plotting modes

## Key Features

### Multiple Setting Paths
The system can run experiments with different FLIMage settings:
- `uncaging_2times.txt`
- `fivepulses.txt` 
- `LTP_by_fivepulses50Hz.txt`

Each setting path gets its own:
- Separate figure/plot
- Unique file naming
- Independent processing

### Flexible Plotting
Two plotting modes available:
- **Single mode**: Each image plotted individually (like original fivepulse script)
- **Multiple mode**: All images from one setting plotted in one figure with tdTomato reference

### Power Calibration
- Automatically loads latest power calibration from JSON files
- **Manual input option**: Users can input slope and intercept values when needed
- Prompts for manual input if JSON files are unavailable or too old
- Configurable option to always use manual input

### Easy Configuration
All parameters are in `titration_config.py`:
- Modify laser power settings
- Change setting file paths
- Adjust plotting parameters
- Set timing intervals

## Usage

### Basic Usage
```python
# Run the main experiment with all setting paths
python GCaMP_unc_combined_titration.py
```

### Single Setting Experiment
```python
# Uncomment in the script to run with only one setting
run_single_setting_experiment()
```

### Custom Configuration
1. Edit `titration_config.py` to modify parameters
2. Run the main script

## Power Calibration Options

### Automatic JSON Calibration (Default)
The system automatically loads power calibration from JSON files in the powermeter directory.

### Manual Power Calibration
Users can input power calibration values manually in several scenarios:

#### Option 1: Force Manual Input
Set in `titration_config.py`:
```python
USE_MANUAL_POWER_CALIBRATION = True
```

#### Option 2: Prompt When JSON is Old
When JSON calibration is more than 24 hours old, the system will ask:
```
Power calibration is more than 24 hours ago
Continue with JSON calibration or use manual input? (json/manual):
```

#### Option 3: Automatic Fallback
If JSON files are unavailable or corrupted, the system automatically prompts for manual input.

### Manual Input Process
When manual input is needed, the system will prompt:
```
=== Manual Power Calibration Input ===
Please enter the power calibration values:
Enter power slope (e.g., 0.158): 
Enter power intercept (e.g., 0.139): 
```

## Configuration Examples

### Change Laser Power Settings
```python
# In titration_config.py
LASER_MW_MS = [
    [2.0, 6],
    [2.4, 6],
    [2.8, 6],
    [3.3, 6],
    [4.0, 6],
    [5.0, 6],  # Add more power levels
]
```

### Modify Setting Paths
```python
# In titration_config.py
SETTINGPATH_LIST = [
    r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\my_custom_setting.txt",
    # Add or remove setting paths
]
```

### Adjust Plotting Parameters
```python
# In titration_config.py
VMIN = 1    # Minimum F/F0 value for color scale
VMAX = 8    # Maximum F/F0 value for color scale
```

### Force Manual Power Calibration
```python
# In titration_config.py
USE_MANUAL_POWER_CALIBRATION = True  # Always prompt for manual input
```

## Output

The system generates:
1. **Individual plots**: Each image saved as PNG in `plot/` folder
2. **Combined plots**: Multiple images in one figure (when using multiple mode)
3. **Console output**: Progress updates and power settings used

## File Organization

```
controlFLIMage/ForUse/
├── flim_analysis_utils.py          # Utility functions
├── titration_config.py             # Configuration parameters
├── GCaMP_unc_combined_titration.py # Main script
└── README_combined_titration.md    # This file
```

## Differences from Original Scripts

### Original `GCaMP_unc_fivepulse_titration_20250612.py`:
- Single setting path
- Individual image plotting
- Hardcoded parameters
- Manual power calibration only

### Original `GCaMP_unc_pow_dur_titration_20250618.py`:
- Single setting path  
- Multiple image plotting in one figure
- JSON power calibration only

### Combined System:
- Multiple setting paths
- Both plotting modes available
- **Flexible power calibration**: JSON + manual input options
- Modular, configurable design
- Better error handling
- Reusable functions

## Troubleshooting

### Power Calibration Issues
- **JSON files missing**: System will automatically prompt for manual input
- **JSON format errors**: System will automatically prompt for manual input
- **Old calibration**: System will ask if you want to use JSON or manual input
- **Force manual input**: Set `USE_MANUAL_POWER_CALIBRATION = True` in config

### File Path Issues
- Ensure all paths in `titration_config.py` are correct
- Check that FLIMage setting files exist
- Verify write permissions for plot output

### Import Errors
- Make sure `sys.path.append("../")` is working
- Check that all required modules are available
- Verify file structure matches expected layout

### Manual Input Errors
- Enter valid numbers for slope and intercept
- Common values: slope ~0.158, intercept ~0.139
- System will keep prompting until valid input is received 