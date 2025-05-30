# Flexible File Selection GUI

The File Selection GUI has been enhanced to be flexible and customizable. You can now specify additional columns to display in the GUI beyond the standard ones.

## Standard Columns (Always Shown)

The GUI always shows these standard columns:

1. **TIFF Path** - Filename of the TIFF file (immutable)
2. **Reject** - Checkbox to mark entries for rejection
3. **Comment** - Text field for comments
4. **z_from, z_to, len_z** - Z-range information (auto-detected if present)
5. **ROI Status** - Spine, DendriticShaft, Background ROI status and dates
6. **Action Buttons** - "All ROIs" and individual ROI buttons

## Adding Custom Columns

You can display additional columns by passing the `additional_columns` parameter:

```python
from gui_roi_analysis.file_selection_gui import launch_file_selection_gui

# Basic usage (no additional columns)
gui = launch_file_selection_gui(combined_df, df_save_path_2)

# With additional columns
additional_columns = [
    'relative_time_min',
    'nth_omit_induction', 
    'phase',
    'corrected_uncaging_x',
    'corrected_uncaging_y'
]

gui = launch_file_selection_gui(
    combined_df, 
    df_save_path_2, 
    additional_columns=additional_columns
)
```

## Column Layout

The columns are displayed in this order:

1. TIFF Path
2. Reject
3. Comment
4. z_from, z_to, len_z (if present)
5. **Your additional columns (inserted here)**
6. ROI status columns
7. Action buttons

## Features

- **Automatic column detection**: Only columns that exist in your DataFrame will be shown
- **Error handling**: Missing columns are ignored with a warning message
- **Immutable display**: Additional columns are read-only by default
- **Tooltips**: Each cell shows the column name and value on hover
- **Smart resizing**: Column widths are automatically adjusted

## Common Use Cases

### Timing Information
```python
timing_columns = [
    'relative_time_min',
    'relative_time_sec',
    'nth_omit_induction',
    'phase'
]
```

### Position Information
```python
position_columns = [
    'corrected_uncaging_x',
    'corrected_uncaging_y',
    'corrected_uncaging_z'
]
```

### Analysis Results
```python
analysis_columns = [
    'intensity_mean',
    'intensity_std',
    'snr_ratio',
    'norm_intensity'
]
```

### Experimental Metadata
```python
metadata_columns = [
    'experimenter',
    'temperature_c',
    'ph_value',
    'drug_concentration'
]
```

## Demo Script

Run the demo script to see examples:

```bash
python gui_roi_analysis/flexible_gui_demo.py
```

This will show you different configurations and help you understand how the flexible GUI works.

## Tips

1. **Start simple**: Begin with just a few important columns
2. **Check your data**: Make sure the columns you want exist in your DataFrame
3. **Use meaningful names**: Column names are displayed as headers
4. **Consider screen space**: Too many columns might make the GUI too wide
5. **Group related columns**: Timing, position, and analysis columns work well together

## Integration with Existing Code

Update your existing code by simply adding the `additional_columns` parameter:

```python
# Before (old way)
file_selection_gui = launch_file_selection_gui(combined_df, df_save_path_2)

# After (new flexible way)
useful_columns = ['relative_time_min', 'phase', 'corrected_uncaging_x']
file_selection_gui = launch_file_selection_gui(
    combined_df, 
    df_save_path_2, 
    additional_columns=useful_columns
)
```

The GUI remains fully backward compatible - existing code will work unchanged. 