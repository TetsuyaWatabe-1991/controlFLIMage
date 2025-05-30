import sys
import os

# Get the absolute path to the Git directory (parent of controlFLIMage)
current_dir = os.getcwd()  # This is controlFLIMage/AnalysisForFLIMage
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # This goes up to Git directory
sys.path.append(parent_dir)

# Now we can import from controlFLIMage
from controlFLIMage.AnalysisForFLIMage.uncaging_pos_multiple import get_uncaging_pos_multiple 