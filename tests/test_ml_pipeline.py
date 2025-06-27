# Add these imports at the top of the file
import os
import gc
import psutil
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import pairwise_distances

# Add this function to your pipeline.py file, before the PhylogeneticPipeline class

# In the PhylogeneticPipeline class, modify the run method:


    
    # Return results dictionary as before
    # ...