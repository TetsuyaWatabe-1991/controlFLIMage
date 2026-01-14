"""
Custom matplotlib.pyplot wrapper that automatically handles plt.show() behavior
based on the execution environment.

In interactive environments (IPython/Jupyter), plt.show() displays plots normally.
In terminal environments, plt.show() is replaced with plt.close(); plt.clf() to avoid popup windows.
"""

import sys
from matplotlib import pyplot as _plt

# Check if running in interactive environment (IPython/Jupyter)
def _is_interactive_environment():
    """Check if running in interactive environment like IPython or Jupyter."""
    try:
        # Method 1: Check for IPython get_ipython function
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                return True
        except (ImportError, NameError):
            pass
        
        # Method 2: Check for ipykernel (Jupyter/VS Code interactive window)
        if 'ipykernel' in sys.modules:
            return True
            
        # Method 3: Check for zmqshell (Jupyter notebook)
        if 'ipykernel.zmqshell' in sys.modules:
            return True
        
        # Method 4: Check if __IPYTHON__ is defined
        try:
            if '__IPYTHON__' in globals() or '__IPYTHON__' in sys.modules:
                return True
        except:
            pass
            
    except Exception:
        pass
    
    return False

# Store original show function
_original_show = _plt.show

def _smart_show(*args, **kwargs):
    """
    Smart show function that displays plots in interactive environments
    but closes them in terminal environments.
    """
    if _is_interactive_environment():
        # In interactive environment, use normal show
        return _original_show(*args, **kwargs)
    else:
        # In terminal environment, close and clear instead
        _plt.close()
        _plt.clf()

# Replace show function
_plt.show = _smart_show

# Export all matplotlib.pyplot functionality
# This allows: from custom_plot import plt
# and use it exactly like: from matplotlib import pyplot as plt
plt = _plt
