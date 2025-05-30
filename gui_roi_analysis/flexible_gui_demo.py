#!/usr/bin/env python3
"""
Demonstration of the flexible File Selection GUI
This script shows how to use the additional_columns parameter to customize
which columns are displayed in the GUI.
"""

import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui_roi_analysis.file_selection_gui import launch_file_selection_gui

def create_demo_dataframe():
    """Create a demo DataFrame with various columns to demonstrate the flexible GUI"""
    
    # Create sample data
    data = {
        'group': ['Group1', 'Group1', 'Group2', 'Group2', 'Group3'],
        'nth_set_label': [0, 1, 0, 1, 0],
        'after_align_save_path': [
            'test1.tiff', 'test2.tiff', 'test3.tiff', 'test4.tiff', 'test5.tiff'
        ],
        
        # Standard z columns (will be auto-detected)
        'z_from': [5, 3, 7, 4, 6],
        'z_to': [15, 13, 17, 14, 16],
        'len_z': [10, 10, 10, 10, 10],
        
        # Timing information
        'relative_time_min': [0.0, 5.5, 10.2, 15.8, 20.1],
        'relative_time_sec': [0.0, 330.0, 612.0, 948.0, 1206.0],
        'nth_omit_induction': [0, 1, 2, 3, 4],
        'phase': ['pre', 'pre', 'post', 'post', 'post'],
        
        # Position information  
        'corrected_uncaging_x': [64.2, 65.1, 63.8, 64.9, 64.5],
        'corrected_uncaging_y': [64.7, 63.9, 65.2, 64.1, 64.8],
        'corrected_uncaging_z': [10, 8, 12, 9, 11],
        
        # Analysis results
        'intensity_mean': [1245.3, 1389.7, 1456.2, 1523.8, 1478.9],
        'intensity_std': [123.4, 145.6, 167.8, 189.2, 156.7],
        'snr_ratio': [10.1, 9.5, 8.7, 8.1, 9.4],
        
        # ROI mask columns (some with data, some without)
        'Spine_roi_mask': [None, np.ones((10, 10)), None, np.ones((8, 8)), None],
        'Spine_roi_analysis_timestamp': [None, '2024-01-01 12:00:00', None, '2024-01-02 14:30:00', None],
        'DendriticShaft_roi_mask': [None, None, None, np.ones((12, 12)), None],
        'DendriticShaft_roi_analysis_timestamp': [None, None, None, '2024-01-02 15:00:00', None],
        'Background_roi_mask': [None, np.ones((15, 15)), None, None, None],
        'Background_roi_analysis_timestamp': [None, '2024-01-01 12:30:00', None, None, None],
        
        # Additional experiment metadata
        'temperature_c': [37.0, 37.2, 36.8, 37.1, 37.0],
        'ph_value': [7.4, 7.4, 7.3, 7.4, 7.4],
        'experimenter': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie'],
    }
    
    return pd.DataFrame(data)

def demo_basic_gui():
    """Demo 1: Basic GUI with no additional columns (just like before)"""
    print("\n=== Demo 1: Basic GUI (no additional columns) ===")
    
    df = create_demo_dataframe()
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    gui = launch_file_selection_gui(df)
    gui.setWindowTitle("Demo 1: Basic GUI")
    gui.show()
    
    return gui, app

def demo_timing_columns():
    """Demo 2: GUI with timing-related columns"""
    print("\n=== Demo 2: GUI with timing columns ===")
    
    df = create_demo_dataframe()
    
    # Define timing-related columns to show
    timing_columns = [
        'relative_time_min',
        'nth_omit_induction', 
        'phase'
    ]
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    gui = launch_file_selection_gui(
        df, 
        additional_columns=timing_columns
    )
    gui.setWindowTitle("Demo 2: GUI with Timing Columns")
    gui.show()
    
    return gui, app

def demo_position_columns():
    """Demo 3: GUI with position-related columns"""
    print("\n=== Demo 3: GUI with position columns ===")
    
    df = create_demo_dataframe()
    
    # Define position-related columns to show
    position_columns = [
        'corrected_uncaging_x',
        'corrected_uncaging_y', 
        'corrected_uncaging_z'
    ]
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    gui = launch_file_selection_gui(
        df, 
        additional_columns=position_columns
    )
    gui.setWindowTitle("Demo 3: GUI with Position Columns")
    gui.show()
    
    return gui, app

def demo_analysis_columns():
    """Demo 4: GUI with analysis result columns"""
    print("\n=== Demo 4: GUI with analysis columns ===")
    
    df = create_demo_dataframe()
    
    # Define analysis-related columns to show
    analysis_columns = [
        'intensity_mean',
        'intensity_std',
        'snr_ratio',
        'temperature_c',
        'experimenter'
    ]
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    gui = launch_file_selection_gui(
        df, 
        additional_columns=analysis_columns
    )
    gui.setWindowTitle("Demo 4: GUI with Analysis Columns")
    gui.show()
    
    return gui, app

def demo_comprehensive():
    """Demo 5: GUI with many additional columns"""
    print("\n=== Demo 5: Comprehensive GUI (many columns) ===")
    
    df = create_demo_dataframe()
    
    # Define a comprehensive set of columns to show
    comprehensive_columns = [
        'relative_time_min',
        'nth_omit_induction',
        'phase',
        'corrected_uncaging_x',
        'corrected_uncaging_y',
        'intensity_mean',
        'snr_ratio',
        'experimenter'
    ]
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    gui = launch_file_selection_gui(
        df, 
        additional_columns=comprehensive_columns
    )
    gui.setWindowTitle("Demo 5: Comprehensive GUI")
    gui.show()
    
    return gui, app

if __name__ == "__main__":
    print("Flexible File Selection GUI Demo")
    print("=" * 40)
    
    # Ask user which demo to run
    print("\nAvailable demos:")
    print("1. Basic GUI (no additional columns)")
    print("2. GUI with timing columns")
    print("3. GUI with position columns") 
    print("4. GUI with analysis result columns")
    print("5. Comprehensive GUI (many columns)")
    print("6. All demos (opens multiple windows)")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        gui, app = demo_basic_gui()
        app.exec_()
    elif choice == "2":
        gui, app = demo_timing_columns()
        app.exec_()
    elif choice == "3":
        gui, app = demo_position_columns()
        app.exec_()
    elif choice == "4":
        gui, app = demo_analysis_columns()
        app.exec_()
    elif choice == "5":
        gui, app = demo_comprehensive()
        app.exec_()
    elif choice == "6":
        # Show all demos
        gui1, app = demo_basic_gui()
        gui2, _ = demo_timing_columns()
        gui3, _ = demo_position_columns()
        gui4, _ = demo_analysis_columns()
        gui5, _ = demo_comprehensive()
        
        # Arrange windows so they don't overlap completely
        gui1.move(100, 100)
        gui2.move(200, 150)
        gui3.move(300, 200)
        gui4.move(400, 250)
        gui5.move(500, 300)
        
        app.exec_()
    else:
        print("Invalid choice. Please run the script again and select 1-6.")
        sys.exit(1) 