# Configuration file for GCaMP uncaging titration experiments

# File paths
DIRECTION_INI = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
POWERMETER_DIR = r"C:\Users\yasudalab\Documents\Tetsuya_Imaging\powermeter"

# Power conversion factor
FROM_THORLAB_TO_COHERENT_FACTOR = 1/3

# Laser power titration settings (mW, ms)
LASER_MW_MS = [
    [2.0, 6],
    [2.4, 6],
    [2.8, 6],
    [3.3, 6],
    [4.0, 6],
    # [5.0, 6],
    # [6.5, 6],
]

# Multiple setting paths for different experiments
SETTINGPATH_LIST = [
    r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\uncaging_2times.txt",
    r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\fivepulses.txt",
    r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\LTP_by_fivepulses50Hz.txt"
]

# Plotting parameters
VMIN = 1
VMAX = 6

# Acquisition parameters
UNCAGING_EVERY_SEC = 30
SEND_COUNTER = True
SEND_BASENAME = True

# Power calibration settings
MAX_POWER_PERCENTAGE = 100
POWER_CALIBRATION_WARNING_HOURS = 24

# Manual power calibration settings
# Set to True if you want to use manual input instead of JSON files
USE_MANUAL_POWER_CALIBRATION = False

# Experiment naming
EXPERIMENT_BASENAME_PREFIX = "titration_"

# Plotting modes
PLOT_TYPE_MULTIPLE = 'multiple'  # Plot all images in one figure
PLOT_TYPE_SINGLE = 'single'      # Plot each image individually

# Wait times
WAIT_BETWEEN_SETTINGS_SEC = 13
WAIT_BETWEEN_ACQUISITIONS_SEC = 30 