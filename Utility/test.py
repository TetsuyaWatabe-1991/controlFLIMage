#%%
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Sine Wave Plot')
plt.grid(True)
plt.legend()

# Display the plot
plt.show()
# %%
