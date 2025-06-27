import time
import psutil
import matplotlib.pyplot as plt
import subprocess
import sys
import threading
import numpy as np

# Data storage
timestamps = []
memory_usage = []
running = True

def monitor_memory():
    while running:
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        timestamps.append(time.time())
        memory_usage.append(mem)
        time.sleep(1)

# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_memory)
monitor_thread.daemon = True
monitor_thread.start()

try:
    # Run your script
    cmd = sys.argv[1:]
    if not cmd:
        cmd = ["python", "tests/test_ml_full_pipeline.py"]
    subprocess.run(cmd)
finally:
    running = False
    monitor_thread.join(timeout=1)
    
    # Plot memory usage
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(timestamps) - timestamps[0], memory_usage)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage Over Time')
    plt.grid(True)
    plt.savefig('memory_usage.png')
    print(f"Peak memory usage: {max(memory_usage):.2f} GB")