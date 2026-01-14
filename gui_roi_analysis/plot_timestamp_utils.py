"""
Utility functions for managing plot timestamps using INI files.
"""

import os
import datetime
import configparser


def get_timestamp_ini_path(filepath_without_number):
    """Get path for timestamp INI file for a filepath group"""
    directory = os.path.dirname(filepath_without_number)
    filename = os.path.basename(filepath_without_number)
    ini_filename = f"{filename}_plot_timestamps.ini"
    print(f"ini_save_path: {os.path.join(directory, ini_filename)}")
    return os.path.join(directory, ini_filename)


def save_plot_timestamp_to_ini(filepath_without_number, plot_type):
    """Save current timestamp to INI file for this filepath group"""
    ini_path = get_timestamp_ini_path(filepath_without_number)

    config = configparser.ConfigParser()
    if os.path.exists(ini_path):
        config.read(ini_path)

    if 'plot_timestamps' not in config:
        config['plot_timestamps'] = {}

    current_time = datetime.datetime.now().isoformat()
    config['plot_timestamps'][f'{plot_type}_last_generated'] = current_time

    with open(ini_path, 'w') as configfile:
        config.write(configfile)

    return ini_path


def should_regenerate_group_plots(each_group_df, plot_type, roi_types):
    """Check if plots should be regenerated for this filepath group"""
    filepath_without_number = each_group_df['filepath_without_number'].iloc[0]
    ini_path = get_timestamp_ini_path(filepath_without_number)
    print("start analyzing timestamp")
    # If INI file doesn't exist, regenerate
    if not os.path.exists(ini_path):
        return True, "No timestamp INI file found"

    try:
        config = configparser.ConfigParser()
        config.read(ini_path)

        # Check if this plot type has been generated before
        timestamp_key = f'{plot_type}_last_generated'
        if 'plot_timestamps' not in config or timestamp_key not in config['plot_timestamps']:
            return True, f"No {plot_type} timestamp found in INI"

        # Get the last generation time
        last_generated_str = config['plot_timestamps'][timestamp_key]
        last_generated = datetime.datetime.fromisoformat(last_generated_str)

        # Get the latest ROI analysis timestamp from this group
        roi_timestamps = []
        for roi_type in roi_types:
            timestamp_col = f"{roi_type}_roi_analysis_timestamp"
            if timestamp_col in each_group_df.columns:
                valid_timestamps = each_group_df[timestamp_col].dropna()
                if len(valid_timestamps) > 0:
                    # Ensure all timestamps are datetime objects
                    for ts in valid_timestamps:
                        if isinstance(ts, str):
                            roi_timestamps.append(datetime.datetime.fromisoformat(ts))
                        elif isinstance(ts, datetime.datetime):
                            roi_timestamps.append(ts)

        if not roi_timestamps:
            return False, "No ROI timestamps found - skipping"

        # Find the latest ROI timestamp
        latest_roi_timestamp = max(roi_timestamps)

        print(f"  Latest ROI timestamp: {latest_roi_timestamp} (type: {type(latest_roi_timestamp)})")
        print(f"  Last plot generated: {last_generated} (type: {type(last_generated)})")

        # Compare timestamps (both are now datetime objects)
        if latest_roi_timestamp > last_generated:
            return True, f"ROI updated at {latest_roi_timestamp}, plots generated at {last_generated}"
        else:
            return False, f"ROI timestamps are up to date (latest: {latest_roi_timestamp}, plots: {last_generated})"

    except Exception as e:
        return True, f"Error reading INI file: {str(e)}"


def print_simple_summary(total_groups, regenerated_groups, skipped_groups, plot_type):
    """Print simple summary of what was processed"""
    print(f"\n{plot_type.upper()} PLOT SUMMARY:")
    print(f"Total groups: {total_groups}")
    print(f"Regenerated: {regenerated_groups}")
    print(f"Skipped: {skipped_groups}")
    print("-" * 40)
