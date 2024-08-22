import time
import subprocess
import threading

class GPUUsageMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.utilization_data = []
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _monitor(self):
        while self.running:
            utilization = self._get_gpu_utilization()
            self.utilization_data.append(utilization)
            time.sleep(self.interval)

    def _get_gpu_utilization(self):
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'], 
            stdout=subprocess.PIPE
        )
        gpu_utilization = float(result.stdout.decode('utf-8').strip())
        return gpu_utilization

    def average_utilization(self):
        return sum(self.utilization_data) / len(self.utilization_data) if self.utilization_data else 0.0