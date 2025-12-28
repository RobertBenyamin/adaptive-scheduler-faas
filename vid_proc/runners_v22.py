import json
import os
import sys
import signal
import threading
import socket
import numpy as np
import time
import heapq
import warnings
import torch
from torch import nn
from deep_river import regression
from river import compose, preprocessing

# Optimization for multi-process environment
torch.set_num_threads(1)
warnings.filterwarnings('ignore')

current_path = "/app/pythonAction"
TMP_DIR = "/tmp"
processQueue = []

# --- 1. Online LSTM Architecture (PyTorch) ---

class LSTMModule(nn.Module):
    """Standard PyTorch LSTM for Regression"""
    def __init__(self, n_features, hidden_size=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape provided by deep-river: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        # We take the output of the last time step
        return self.fc(out[:, -1, :])

class OnlineLSTMManager:
    """Wrapper using deep-river for Incremental Learning"""
    def __init__(self, seq_len=3):
        # The RollingRegressor maintains the sliding window history internally
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            regression.RollingRegressor(
                module=LSTMModule,
                loss_fn=nn.MSELoss(),
                optimizer=torch.optim.Adam,
                lr=0.01,
                window_size=seq_len,
                n_features=8 # Matches your 8 feature set
            )
        )
        self.samples_learned = 0

    def predict(self, features):
        # If no samples learned yet, return a safe default
        if self.samples_learned < 1:
            return 2.0
        return self.model.predict_one(features)

    def learn(self, features, actual_duration):
        self.model.learn_one(features, actual_duration)
        self.samples_learned += 1

# --- 2. Global Variables & State ---

serverSocket_ = None
actionModule = None
numCores = 8
affinity_mask = {0, 1, 2, 3, 4, 5, 6, 7}

lockPIDMap = threading.Lock()
mapPIDtoStatus = {}  # {pid: status}
processExecutionHistory = {"function_history": []} #
processStartTime = {}
processExecutedTime = {}
processArrivalTime = {}
processTimestamps = {} # {pid: (acc_wait, last_wait_start)}

# Initialize the Online model
model_lock = threading.Lock()
online_model = OnlineLSTMManager(seq_len=3)

# --- 3. Feature Engineering & Prediction ---

def extract_features(history, current_arrival, last_arrival):
    """Identical feature set to your ARF implementation"""
    n = len(history)
    lag_1 = history[-1] if n >= 1 else 0
    lag_2 = history[-2] if n >= 2 else lag_1
    lag_3 = history[-3] if n >= 3 else lag_2
    
    window_5 = history[-5:] if n >= 5 else history
    mean_5 = np.mean(window_5) if n > 0 else 0
    
    window_10 = history[-10:] if n >= 10 else history
    std_10 = np.std(window_10) if len(window_10) >= 2 else 0
    
    return {
        "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
        "mean_5": mean_5, "std_10": std_10,
        "delta_1_2": lag_1 - lag_2,
        "inter_arrival": min(current_arrival - last_arrival, 10.0) if last_arrival > 0 else 0,
        "history_size": min(n, 100)
    }

def get_burst_time_prediction():
    history = processExecutionHistory["function_history"]
    with model_lock:
        # Use simple mean for very early cold-start
        if len(history) < 5:
            return np.mean(history) if history else 2.0
        
        features = extract_features(history, time.time(), getattr(online_model, 'last_arrival', 0))
        pred = online_model.predict(features)
    return max(0.1, min(pred, 60.0))

def calculate_remaining_time(pid):
    """Remaining time calculation from your paper logic"""
    estimated_burst_time = get_burst_time_prediction()
    elapsed_time = processExecutedTime.get(pid, 0)
    if pid in processStartTime:
        elapsed_time += time.time() - processStartTime[pid]
    return max(estimated_burst_time - elapsed_time, 0)

# --- 4. Scheduling & Preemption Framework ---

PREEMPTION_THRESHOLD = 4 #

def waitTermination(childPid):
    """Monitors completion and triggers Online Learning"""
    global processQueue, mapPIDtoStatus
    os.waitpid(childPid, 0)

    lockPIDMap.acquire()
    try:
        mapPIDtoStatus.pop(childPid, None)
        total_executed = processExecutedTime.get(childPid, 0.0)
        if childPid in processStartTime:
            total_executed += time.time() - processStartTime[childPid]

        arrival = processArrivalTime.get(childPid, time.time())
        history_context = list(processExecutionHistory["function_history"])
        
        # ONLINE LEARNING STEP
        with model_lock:
            features = extract_features(history_context, arrival, getattr(online_model, 'last_arrival', 0))
            online_model.learn(features, total_executed)
            online_model.last_arrival = arrival

        processExecutionHistory["function_history"].append(total_executed)
        if len(processExecutionHistory["function_history"]) > 100:
            processExecutionHistory["function_history"].pop(0)

        # Cleanup
        processTimestamps.pop(childPid, None)
        processStartTime.pop(childPid, None)
        processExecutedTime.pop(childPid, None)
        processArrivalTime.pop(childPid, None)
    except Exception as e:
        print(f"Termination Error: {e}")

    # --- Preemption & Next Task Selection ---
    if processQueue:
        # Filter dead PIDs
        processQueue[:] = [(rt, p) for rt, p in processQueue if os.path.exists(f"/proc/{p}")]
        heapq.heapify(processQueue)

        if processQueue:
            # Priority logic based on 1/RT + Aging
            def priority_selector(item):
                p_rt = calculate_remaining_time(item[1]) + 1e-9
                acc_w, last_w = processTimestamps.get(item[1], (0.0, None))
                wait = acc_w + (time.time() - last_w if last_w else 0)
                return (0.8 * (1/p_rt)) + (0.3 * wait)

            candidates = sorted(processQueue, key=priority_selector, reverse=True)
            _, next_pid = candidates[0]

            # Preemption logic
            current_running = next((p for p, s in mapPIDtoStatus.items() if s == "running"), None)
            if current_running:
                if calculate_remaining_time(next_pid) < calculate_remaining_time(current_running) - PREEMPTION_THRESHOLD:
                    os.kill(current_running, signal.SIGTSTP)
                    mapPIDtoStatus[current_running] = "waiting"
                    # Accumulate time and push back to queue
                    elapsed = time.time() - processStartTime.pop(current_running)
                    processExecutedTime[current_running] = processExecutedTime.get(current_running, 0) + elapsed
                    processTimestamps[current_running] = (processTimestamps[current_running][0], time.time())
                    heapq.heappush(processQueue, (calculate_remaining_time(current_running), current_running))

            # Resume Next
            if next_pid in mapPIDtoStatus:
                mapPIDtoStatus[next_pid] = "running"
                os.kill(next_pid, signal.SIGCONT)
                processStartTime[next_pid] = time.time()
                acc, last = processTimestamps[next_pid]
                processTimestamps[next_pid] = (acc + (time.time() - last if last else 0), None)
                # Remove from queue
                processQueue[:] = [x for x in processQueue if x[1] != next_pid]
                heapq.heapify(processQueue)

    lockPIDMap.release()

# --- 5. FaaS Boilerplate (Socket, IO, Utils) ---

def handle_client_connection(clientSocket, address):
    global numCores, processQueue
    try:
        data_ = clientSocket.recv(4096)
        dataStr = data_.decode('UTF-8')
        if 'Host' not in dataStr: return clientSocket.close()
        
        message = json.loads(dataStr.splitlines()[-1])
        
        # Admin Commands
        if "Clear" in message:
            processExecutionHistory["function_history"] = []
            with model_lock:
                global online_model
                online_model = OnlineLSTMManager(seq_len=3)
            clientSocket.send(b"HTTP/1.1 200 OK\r\n\r\nHistory Reset")
            return clientSocket.close()

        arrival = time.time()
        lockPIDMap.acquire()
        
        # Scale Check
        running = sum(1 for s in mapPIDtoStatus.values() if s == "running")
        child = os.fork()
        
        if child == 0:
            import app # Local function logic
            result = app.lambda_handler(message)
            msg = json.dumps(result)
            clientSocket.send(f"HTTP/1.1 200 OK\r\nContent-Length: {len(msg)}\r\n\r\n{msg}".encode())
            os._exit(0)
        else:
            processArrivalTime[child] = arrival
            if running >= numCores:
                mapPIDtoStatus[child] = "waiting"
                os.kill(child, signal.SIGTSTP)
                processTimestamps[child] = (0.0, time.time())
                heapq.heappush(processQueue, (calculate_remaining_time(child), child))
            else:
                mapPIDtoStatus[child] = "running"
                processStartTime[child] = time.time()
                processTimestamps[child] = (0.0, None)
            
            lockPIDMap.release()
            threading.Thread(target=waitTermination, args=(child,)).start()

    except Exception as e:
        clientSocket.close()

def run():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 8081))
    server.listen(10)
    print("Online-LSTM Runner Active on Port 8081")

    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client_connection, args=(conn, addr)).start()

if __name__ == "__main__":
    run()