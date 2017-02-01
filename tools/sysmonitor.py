import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import psutil
import threading

from pynvml import (nvmlInit,
                     nvmlDeviceGetCount,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetUtilizationRates,
                     nvmlDeviceGetName)

def gpu_info():
    "Returns a tuple of (GPU ID, GPU Description, GPU % Utilization)"
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    info = []
    for i in range(0, deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        util = nvmlDeviceGetUtilizationRates(handle)
        desc = nvmlDeviceGetName(handle)
        info.append((i, desc, util.gpu)) #['GPU %i - %s' % (i, desc)] = util.gpu
    return info

utils = []

class SysMonitor(threading.Thread):
    shutdown = False

    def __init__(self):
        self.utils = []
        self.start_time = time.time()
        self.duration = 0
        threading.Thread.__init__(self)

    def run(self):
        utils = []
        while not self.shutdown:
            dt = datetime.datetime.now()
            util = gpu_info()
            cpu_percent = psutil.cpu_percent()
            self.utils.append([dt] + [x[2] for x in util] + [cpu_percent])
            time.sleep(.1)

    def stop(self):
        self.shutdown = True
        self.duration = time.time() - self.start_time

    def plot(self, title, vert=False):
        if vert:
            fig, ax = plt.subplots(2, 1, figsize=(15, 6))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, size=24)
        ax[0].title.set_text('GPU Utilization')
        ax[0].plot([u[1] for u in self.utils])
        ax[0].set_ylim([0, 100])
        ax[1].title.set_text('CPU Utilization')
        ax[1].plot([u[2] for u in self.utils])
        ax[1].set_ylim([0, 100])
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
