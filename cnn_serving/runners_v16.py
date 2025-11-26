from collections import deque
import json
import os
import sys
import signal
import threading
import socket
import numpy as np
import time
import signal
import requests
from threading import Thread
from storage_helper import download_file, upload_file
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import heapq

current_path = "/app/pythonAction"
BETA = 0.3  # Weight for wait time
processQueue = []
processStartTime = {}


def signal_handler(sig, frame):
    serverSocket_.close()
    sys.exit(0)


class PrintHook:
    def __init__(self, out=1):
        self.func = None
        self.origOut = None
        self.out = out

    def TestHook(self, text):
        f = open('hook_log.txt', 'a')
        f.write(text)
        f.close()
        return 0, 0, text

    def Start(self, func=None):
        if self.out:
            sys.stdout = self
            self.origOut = sys.__stdout__
        else:
            sys.stderr = self
            self.origOut = sys.__stderr__

        if func:
            self.func = func
        else:
            self.func = self.TestHook

    def Stop(self):
        self.origOut.flush()
        if self.out:
            sys.stdout = sys.__stdout__
        else:
            sys.stderr = sys.__stderr__
        self.func = None

    def flush(self):
        self.origOut.flush()

    def write(self, text):
        proceed = 1
        lineNo = 0
        addText = ''
        if self.func != None:
            proceed, lineNo, newText = self.func(text)
        if proceed:
            if text.split() == []:
                self.origOut.write(text)
            else:
                if self.out:
                    if lineNo:
                        try:
                            raise "Dummy"
                        except:
                            codeObject = sys.exc_info(
                            )[2].tb_frame.f_back.f_code
                            fileName = codeObject.co_filename
                            funcName = codeObject.co_name
                self.origOut.write(newText)


def MyHookOut(text):
    return 1, 1, ' -- pid -- ' + str(os.getpid()) + ' ' + text


# Global variables
serverSocket_ = None  # serverSocket
actionModule = None  # action module

checkTable = {}
mapPIDtoLeader = {}
checkTableShadow = {}
valueTable = {}
mapPIDtoIO = {}
lockCache = threading.Lock()

processTimestamps = {}  # {pid: (initial_burst, start_time)}
FUNCTION_HISTORY_KEY = "function_history"
# Menyimpan histori eksekusi proses
processExecutionHistory = {FUNCTION_HISTORY_KEY: []}
processStartTime = {}

lockPIDMap = threading.Lock()
requestQueue = []  # queue of child processes
mapPIDtoStatus = {}  # map from pid to status (running, waiting)

processArrivalTimes = {}  # Dictionary to track arrival times of processes
responseMapWindows = []  # map from pid to response

affinity_mask = {0, 1, 2, 3, 4, 5, 6, 7}

# Prediction accuracy tracking
prediction_errors = {
    "ewma": deque(maxlen=20),
    "rf": deque(maxlen=20),
    "trend": deque(maxlen=20)
}

# Adaptive weights (updated dynamically)
adaptive_weights = {
    "ewma": 0.33,
    "rf": 0.33,
    "trend": 0.34
}

# Model cache
rf_model_cache = None
rf_model_lock = threading.Lock()

# The function to update the core nums by request.


def updateThread():
    # Shared vaiable: numCores
    global numCores

    # Bind to 0.0.0.0:5500
    myHost = '0.0.0.0'
    myPort = 5500

    # Create a socket
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind((myHost, myPort))
    serverSocket.listen(1)

    # Handle request
    while True:
        # Accept a connection
        (clientSocket, _) = serverSocket.accept()
        data_ = clientSocket.recv(1024)
        dataStr = data_.decode('UTF-8')
        dataStrList = dataStr.splitlines()
        message = json.loads(dataStrList[-1])

        # Get the numCores and update the global variable
        numCores = message["numCores"]
        result = {"Response": "Ok"}
        msg = json.dumps(result)

        # Send the result and close the socket
        response_headers = {
            'Content-Type': 'text/html; encoding=utf8',
            'Content-Length': len(msg),
            'Connection': 'close',
        }

        response_headers_raw = ''.join('%s: %s\r\n' % (
            k, v) for k, v in response_headers.items())

        response_proto = 'HTTP/1.1'
        response_status = '200'
        response_status_text = 'OK'

        r = '%s %s %s\r\n' % (
            response_proto, response_status, response_status_text)
        print(r)

        clientSocket.send(r.encode(encoding="utf-8"))
        clientSocket.send(response_headers_raw.encode(encoding="utf-8"))
        clientSocket.send('\r\n'.encode(encoding="utf-8"))
        clientSocket.send(msg.encode(encoding="utf-8"))

        clientSocket.close()


def myFunction(data_, clientSocket_):
    # Measure the start time for burst time calculation
    startTime = time.time()

    global actionModule
    global numCores

    dataStr = data_.decode('UTF-8')
    dataStrList = dataStr.splitlines()
    numCoreFlag = False
    message = None
    try:
        message = json.loads(dataStrList[-1])
        numCores = int(message["numCores"])
        numCoreFlag = True
        result = {"Response": "Ok"}
        msg = json.dumps(result)
    except:
        pass

    # Set the main function
    if numCoreFlag == False:
        result = actionModule.lambda_handler(message)

        # Send the result (Test Pid)
        result["myPID"] = os.getpid()
        msg = json.dumps(result)

    response_headers = {
        'Content-Type': 'text/html; encoding=utf8',
        'Content-Length': len(msg),
        'Connection': 'close',
    }

    response_headers_raw = ''.join('%s: %s\r\n' % (k, v)
                                   for k, v in response_headers.items())

    response_proto = 'HTTP/1.1'
    response_status = '200'
    response_status_text = 'OK'  # this can be random

    # sending all this stuff
    r = '%s %s %s\r\n' % (response_proto, response_status,
                          response_status_text)
    try:
        clientSocket_.send(r.encode(encoding="utf-8"))
        clientSocket_.send(response_headers_raw.encode(encoding="utf-8"))
        # to separate headers from body
        clientSocket_.send('\r\n'.encode(encoding="utf-8"))
        clientSocket_.send(msg.encode(encoding="utf-8"))
    except:
        clientSocket_.close()
    clientSocket_.close()

    # Measure the end time for burst time calculation
    endTime = time.time()

    # Return the measured burst time (execution time)
    burstTime = endTime - startTime
    return burstTime

# Fungsi EWMA (Exponential Weighted Moving Average)


def calculate_ewma(history, alpha=0.8):
    """EWMA predictor"""
    if not history:
        return 0
    ewma = history[0]
    for val in history[1:]:
        ewma = alpha * val + (1 - alpha) * ewma
    return ewma


def calculate_trend_predictor(history):
    """
    Trend-based predictor using linear extrapolation.
    Novel contribution: Uses weighted recent history.
    """
    if len(history) < 3:
        return np.mean(history)

    # Use last 10 samples for trend
    recent = history[-10:] if len(history) >= 10 else history

    # Weighted linear regression (more weight on recent samples)
    weights = np.exp(np.linspace(-1, 0, len(recent)))
    weights /= weights.sum()

    X = np.arange(len(recent))
    y = np.array(recent)

    # Weighted mean
    mean_x = np.sum(weights * X)
    mean_y = np.sum(weights * y)

    # Weighted slope
    numerator = np.sum(weights * (X - mean_x) * (y - mean_y))
    denominator = np.sum(weights * (X - mean_x) ** 2)

    if denominator == 0:
        return recent[-1]

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    # Predict next value
    prediction = slope * len(recent) + intercept

    return max(prediction, 0)


def train_rf_lightweight(history):
    """
    Lightweight Random Forest with feature engineering.
    """
    if len(history) < 10:
        return None

    window_size = 5
    X, y = [], []

    for i in range(window_size, len(history)):
        window = history[i-window_size:i]
        features = [
            np.mean(window),
            np.std(window),
            np.max(window),
            np.min(window),
            window[-1],
            (window[-1] - window[0]) / window_size,
        ]
        X.append(features)
        y.append(history[i])

    if len(X) < 5:
        return None

    X = np.array(X)
    y = np.array(y)

    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=4,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    return model


def predict_rf(model, history):
    """Make RF prediction"""
    if model is None or len(history) < 5:
        return None

    window = history[-5:]
    features = np.array([[
        np.mean(window),
        np.std(window),
        np.max(window),
        np.min(window),
        window[-1],
        (window[-1] - window[0]) / 5,
    ]])

    return max(float(model.predict(features)[0]), 0)


def calculate_prediction_confidence(history, predictions):
    """
    Novel: Calculate confidence based on prediction variance and recent accuracy.
    """
    if len(predictions) < 2:
        return 0.5  # Default confidence

    # Variance in predictions (low variance = high confidence)
    pred_std = np.std(list(predictions.values()))
    pred_mean = np.mean(list(predictions.values()))

    if pred_mean == 0:
        return 0.5

    coefficient_of_variation = pred_std / pred_mean

    # Inverse relationship: lower CV = higher confidence
    confidence = 1.0 / (1.0 + coefficient_of_variation)

    return min(max(confidence, 0.1), 0.9)


def update_adaptive_weights(actual_time):
    """
    Novel: Update predictor weights based on recent accuracy.
    Methods that perform better get higher weights.
    """
    global adaptive_weights

    # Calculate recent accuracy (inverse of error)
    accuracies = {}
    for method in prediction_errors:
        if len(prediction_errors[method]) > 0:
            # Mean Absolute Percentage Error (MAPE)
            mape = np.mean([abs(err) for err in prediction_errors[method]])
            accuracies[method] = 1.0 / (1.0 + mape)
        else:
            accuracies[method] = 0.33

    # Normalize to sum to 1.0
    total = sum(accuracies.values())
    if total > 0:
        adaptive_weights = {k: v/total for k, v in accuracies.items()}

    # Add momentum (smooth weight changes)
    momentum = 0.7
    for method in adaptive_weights:
        if method in accuracies:
            adaptive_weights[method] = (
                momentum * adaptive_weights[method] +
                (1 - momentum) * (accuracies[method] / total)
            )


def detect_system_load():
    """
    Detect system load state: LOW, MEDIUM, HIGH.
    Based on queue length and recent execution times.
    """
    global processQueue

    queue_length = len(processQueue)

    if queue_length == 0:
        return "LOW"
    elif queue_length < 5:
        return "MEDIUM"
    else:
        return "HIGH"

# ===== MAIN PREDICTION FUNCTION (AWHP) =====


def calculate_remaining_time_awhp(pid):
    """
    Adaptive Weighted Hybrid Predictor (AWHP) - YOUR NOVEL ALGORITHM.

    Key innovations:
    1. Adaptive weighting based on recent prediction accuracy
    2. Confidence-aware prediction
    3. System load-aware adjustment
    4. Error tracking and online learning
    """
    global rf_model_cache, adaptive_weights
    history = processExecutionHistory[FUNCTION_HISTORY_KEY]

    if not history:
        initial_burst, _ = processTimestamps.get(pid, (2, time.time()))
        return initial_burst

    # === STEP 1: Get predictions from all methods ===
    predictions = {}

    # EWMA prediction
    ewma_pred = calculate_ewma(history)
    predictions["ewma"] = ewma_pred

    # Trend prediction
    trend_pred = calculate_trend_predictor(history)
    predictions["trend"] = trend_pred

    # Random Forest prediction
    if len(history) >= 10:
        with rf_model_lock:
            if rf_model_cache is None or len(history) % 15 == 0:
                rf_model_cache = train_rf_lightweight(history)

        if rf_model_cache is not None:
            rf_pred = predict_rf(rf_model_cache, history)
            if rf_pred is not None:
                predictions["rf"] = rf_pred
            else:
                predictions["rf"] = ewma_pred
        else:
            predictions["rf"] = ewma_pred
    else:
        predictions["rf"] = ewma_pred

    # === STEP 2: Calculate prediction confidence ===
    confidence = calculate_prediction_confidence(history, predictions)

    # === STEP 3: Detect system load ===
    system_load = detect_system_load()

    # === STEP 4: Adjust weights based on system load ===
    load_adjusted_weights = adaptive_weights.copy()

    if system_load == "LOW":
        # Trust trend more in low load (stable)
        load_adjusted_weights["trend"] *= 1.2
    elif system_load == "HIGH":
        # Trust EWMA more in high load (responsive)
        load_adjusted_weights["ewma"] *= 1.3

    # Renormalize
    total_weight = sum(load_adjusted_weights.values())
    load_adjusted_weights = {k: v/total_weight for k,
                             v in load_adjusted_weights.items()}

    # === STEP 5: Compute weighted ensemble prediction ===
    ensemble_pred = sum(
        load_adjusted_weights[method] * predictions[method]
        for method in predictions
    )

    # === STEP 6: Confidence-aware adjustment ===
    # If confidence is low, be more conservative (add safety margin)
    if confidence < 0.5:
        std_dev = np.std(history[-10:]) if len(history) >= 10 else 0
        ensemble_pred += 0.3 * std_dev  # Safety margin

    # === STEP 7: Calculate remaining time ===
    elapsed_time = time.time() - processStartTime.get(pid, time.time())
    remaining_time = max(ensemble_pred - elapsed_time, 0)

    return remaining_time


def update_prediction_error(pid, actual_time):
    """
    Track prediction errors for adaptive weight updates.
    """
    history = processExecutionHistory[FUNCTION_HISTORY_KEY]

    if len(history) < 2:
        return

    # Get previous predictions
    ewma_pred = calculate_ewma(history[:-1])
    trend_pred = calculate_trend_predictor(history[:-1])

    # Calculate errors
    prediction_errors["ewma"].append(
        abs(ewma_pred - actual_time) / (actual_time + 1e-9))
    prediction_errors["trend"].append(
        abs(trend_pred - actual_time) / (actual_time + 1e-9))

    # RF error (if model exists)
    if rf_model_cache is not None and len(history) >= 10:
        rf_pred = predict_rf(rf_model_cache, history[:-1])
        if rf_pred is not None:
            prediction_errors["rf"].append(
                abs(rf_pred - actual_time) / (actual_time + 1e-9))

    # Update weights based on errors
    update_adaptive_weights(actual_time)


def calculate_total_wait_time(processQueue):
    """
    Calculate total wait time for all waiting processes
    """
    total_wait_time = 0
    current_time = time.time()

    for process_item in processQueue:
        _, pid = process_item
        if pid in processTimestamps:
            # Calculate individual wait time
            _, start_time = processTimestamps[pid]
            total_wait_time += (current_time - start_time)

    return total_wait_time


def calculate_dynamic_beta(total_wait_time, num_tasks):
    """
    Calculate dynamic beta based on system-wide wait time characteristics
    """
    if num_tasks == 0:
        return 0.2  # Default fallback value

    # Dynamic beta calculation
    dynamic_beta = total_wait_time / (num_tasks + 1)

    # Normalization to prevent extreme values
    return min(max(dynamic_beta, 0.1), 1.0)


# Batas waktu maksimum sebelum preemption terjadi (dalam detik)
PREEMPTION_THRESHOLD = 4


def waitTermination(childPid):
    """
    Menunggu proses selesai atau menggantinya jika ada proses lebih prioritas dengan preemption.
    """
    global processQueue, mapPIDtoStatus

    os.waitpid(childPid, 0)  # Tunggu hingga proses selesai

    lockPIDMap.acquire()

    try:
        # Hapus proses dari status map
        mapPIDtoStatus.pop(childPid, None)

        # Simpan burst time ke history
        if childPid in processStartTime:
            elapsed = time.time() - processStartTime[childPid]
            processExecutionHistory[FUNCTION_HISTORY_KEY].append(elapsed)

            # Update prediction errors for adaptive learning
            update_prediction_error(childPid, elapsed)

    except Exception as e:
        print(f"Error removing process {childPid}: {e}")

    # PREEMPTION: Cek apakah ada proses dengan waktu tersisa lebih pendek dari proses yang berjalan
    if processQueue:
        # Urutkan queue berdasarkan 1 / remaining_time untuk SRTF
        def priority_selector(process_item):
            _, pid = process_item
            remaining_time = calculate_remaining_time_awhp(pid) + 1e-9

            # Calculate individual wait time
            if pid in processTimestamps:
                _, start_time = processTimestamps[pid]
                individual_wait_time = time.time() - start_time
            else:
                individual_wait_time = 0

             # Calculate dynamic beta
            total_wait_time = calculate_total_wait_time(processQueue)
            dynamic_beta = calculate_dynamic_beta(
                total_wait_time, len(processQueue))

            # Alpha for remaining time (inverse priority)
            alpha = 0.8

            # Priority calculation
            priority = (alpha * (1 / (remaining_time + 1e-9))) + \
                (dynamic_beta * individual_wait_time)

            return priority

        # Ambil proses dengan prioritas tertinggi
        next_process_candidates = sorted(
            processQueue, key=priority_selector, reverse=True)

        if next_process_candidates:
            _, nextProcess = next_process_candidates[0]
            current_running_pid = None

            # Cari proses yang sedang berjalan
            for pid, status in mapPIDtoStatus.items():
                if status == "running":
                    current_running_pid = pid
                    break

            # Jika ada proses yang sedang berjalan, cek apakah harus di-preempt
            if current_running_pid:
                current_remaining = calculate_remaining_time_awhp(
                    current_running_pid)
                next_remaining = calculate_remaining_time_awhp(nextProcess)

                # PREEMPTION CHECK
                if next_remaining < current_remaining - PREEMPTION_THRESHOLD:
                    print(f"Preempting process {current_running_pid} (remaining: {current_remaining:.2f}s) "
                          f"with process {nextProcess} (remaining: {next_remaining:.2f}s)")

                    # Hentikan proses yang berjalan
                    try:
                        os.kill(current_running_pid, signal.SIGSTOP)
                        mapPIDtoStatus[current_running_pid] = "paused"
                    except Exception as e:
                        print(
                            f"Error stopping process {current_running_pid}: {e}")

            # Jalankan proses dengan prioritas tertinggi
            processQueue.remove((_, nextProcess))
            mapPIDtoStatus[nextProcess] = "running"

            try:
                os.kill(nextProcess, signal.SIGCONT)
                # Reset waktu mulai eksekusi
                processStartTime[nextProcess] = time.time()
            except Exception as e:
                print(f"Error resuming process {nextProcess}: {e}")

    lockPIDMap.release()


def performIO(clientSocket_):
    global mapPIDtoStatus
    global numCores
    global checkTable
    global mapPIDtoIO
    global valueTable
    global checkTableShadow
    global mapPIDtoLeader

    data_ = b''
    data_ += clientSocket_.recv(1024)
    dataStr = data_.decode('UTF-8')

    while True:
        dataStrList = dataStr.splitlines()

        message = None
        try:
            message = json.loads(dataStrList[-1])
            break
        except:
            data_ += clientSocket_.recv(1024)
            dataStr = data_.decode('UTF-8')

    operation = message["operation"]
    blobName = message["blobName"]
    blockedID = message["pid"]

    my_id = threading.get_native_id()

    lockPIDMap.acquire()
    mapPIDtoStatus[blockedID] = "blocked"
    for child in mapPIDtoStatus.copy():
        if child in mapPIDtoStatus:
            if mapPIDtoStatus[child] == "waiting":
                mapPIDtoStatus[child] = "running"
                try:
                    os.kill(child, signal.SIGCONT)
                    break
                except:
                    pass
    lockPIDMap.release()

    if operation == "get":
        lockCache.acquire()
        if blobName in checkTable:
            myLeader = mapPIDtoLeader[blobName]
            myEvent = threading.Event()
            mapPIDtoIO[my_id] = myEvent
            checkTable[blobName].append(my_id)
            checkTableShadow[myLeader].append(my_id)
            lockCache.release()
            myEvent.wait()
            lockCache.acquire()
            blob_val = valueTable[myLeader]
            mapPIDtoIO.pop(my_id)
            checkTableShadow[myLeader].remove(my_id)
            if len(checkTableShadow[myLeader]) == 0:
                checkTableShadow.pop(myLeader)
                valueTable.pop(myLeader)
            lockCache.release()
        else:
            mapPIDtoLeader[blobName] = my_id
            checkTable[blobName] = []
            checkTableShadow[my_id] = []
            checkTable[blobName].append(my_id)
            lockCache.release()
            blob_storage = blobName.split("_")[0]
            download_file(blobName, f"{current_path}/{blobName}")
            with open(f"{current_path}/{blobName}", "rb") as file:
                blob_val = file.read()

            lockCache.acquire()
            valueTable[my_id] = blob_val
            checkTable[blobName].remove(my_id)
            for elem in checkTable[blobName]:
                mapPIDtoIO[elem].set()
            checkTable.pop(blobName)
            lockCache.release()

        full_blob_name = blobName.split(".")
        proc_blob_name = full_blob_name[0] + "_" + \
            str(blockedID) + "." + full_blob_name[1]
        with open(proc_blob_name, "wb") as my_blob:
            my_blob.write(blob_val)
    else:
        fReadname = message["value"]
        fRead = open(fReadname, "rb")
        value = fRead.read()
        upload_file(f"{current_path}/{value}", f"files/{blobName}")
        blob_val = "none"

    lockPIDMap.acquire()
    numRunning = 0  # number of running processes
    for child in mapPIDtoStatus.copy():
        if mapPIDtoStatus[child] == "running":
            numRunning += 1
    if numRunning < numCores:
        mapPIDtoStatus[blockedID] = "running"
        os.kill(blockedID, signal.SIGCONT)
    else:
        mapPIDtoStatus[blockedID] = "waiting"
        os.kill(blockedID, signal.SIGSTOP)
    lockPIDMap.release()

    messageToRet = json.dumps({"value": "OK"})
    try:
        os.kill(blockedID, signal.SIGCONT)
    except:
        pass
    clientSocket_.send(messageToRet.encode(encoding="utf-8"))
    try:
        os.kill(blockedID, signal.SIGCONT)
    except:
        pass
    # clientSocket_.close()


def IOThread():
    myHost = '0.0.0.0'
    myPort = 3333

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind((myHost, myPort))
    serverSocket.listen(1)

    while True:
        (clientSocket, _) = serverSocket.accept()
        threading.Thread(target=performIO, args=(clientSocket,)).start()


def run():

    # serverSocket_: socket
    # actionModule:  the module to execute
    # requestQueue:
    # mapPIDtoStatus: store status for each process (waiting / running)
    global serverSocket_
    global actionModule
    global requestQueue
    global mapPIDtoStatus
    global numCores
    global responseMapWindows
    global affinity_mask
    global processQueue
    global processStartTime

    # Set the core of mxcontainer
    numCores = 8
    os.sched_setaffinity(0, affinity_mask)

    print("Welcome... ", numCores)

    # Set the address and port, the port can be acquired from environment variable
    myHost = '0.0.0.0'
    myPort = int(os.environ.get('PORT', 8081))

    # Bind the address and port
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind((myHost, myPort))
    serverSocket.listen(1)

    # serverSocket_ = serverSocket

    # Set actionModule
    import app
    actionModule = app

    # Set the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Redirect the stdOut and stdErr
    phOut = PrintHook()
    phOut.Start(MyHookOut)

    # Monitor numCore update
    threadUpdate = threading.Thread(target=updateThread)
    threadUpdate.start()

    # Monitor I/O Block
    threadIntercept = threading.Thread(target=IOThread)
    threadIntercept.start()

    # If a request come, then fork.
    while (True):

        (clientSocket, address) = serverSocket.accept()
        print("Accept a new connection from %s" % str(address), flush=True)

        data_ = b''

        data_ += clientSocket.recv(1024)

        dataStr = data_.decode('UTF-8')

        if 'Host' not in dataStr:
            msg = 'OK'
            response_headers = {
                'Content-Type': 'text/html; encoding=utf8',
                'Content-Length': len(msg),
                'Connection': 'close',
            }
            response_headers_raw = ''.join('%s: %s\r\n' % (
                k, v) for k, v in response_headers.items())

            response_proto = 'HTTP/1.1'
            response_status = '200'
            response_status_text = 'OK'  # this can be random

            # sending all this stuff
            r = '%s %s %s\r\n' % (
                response_proto, response_status, response_status_text)
            try:
                clientSocket.send(r.encode(encoding="utf-8"))
                clientSocket.send(
                    response_headers_raw.encode(encoding="utf-8"))
                # to separate headers from body
                clientSocket.send('\r\n'.encode(encoding="utf-8"))
                clientSocket.send(msg.encode(encoding="utf-8"))
                clientSocket.close()
                continue
            except:
                clientSocket.close()
                continue

        while True:
            dataStrList = dataStr.splitlines()

            message = None
            try:
                message = json.loads(dataStrList[-1])
                break
            except:
                data_ += clientSocket.recv(1024)
                dataStr = data_.decode('UTF-8')

        responseFlag = False
        if message != None:

            if "numCores" in message:
                numCores = int(message["numCores"])
                result = {"Response": "Ok"}
                responseMapWindows = []
                if "affinity_mask" in message:
                    affinity_mask = message["affinity_mask"]
                    os.sched_setaffinity(0, affinity_mask)
                msg = json.dumps(result)
                responseFlag = True

            if "Q" in message:
                i = []
                for responseTime in responseMapWindows:
                    if responseTime[1][1] != -1:
                        i.append(responseTime[1][1] - responseTime[1][0])
                if len(i) == 0:
                    result = {"p95": 0}
                else:
                    result = {"p95": np.percentile(i, 95)}
                result["affinity_mask"] = list(affinity_mask)
                result["numCores"] = numCores
                msg = json.dumps(result)
                responseFlag = True

            if "Clear" in message:
                responseMapWindows = []

        if responseFlag == True:
            response_headers = {
                'Content-Type': 'text/html; encoding=utf8',
                'Content-Length': len(msg),
                'Connection': 'close',
            }
            response_headers_raw = ''.join('%s: %s\r\n' % (
                k, v) for k, v in response_headers.items())

            response_proto = 'HTTP/1.1'
            response_status = '200'
            response_status_text = 'OK'  # this can be random

            # sending all this stuff
            r = '%s %s %s\r\n' % (
                response_proto, response_status, response_status_text)

            clientSocket.send(r.encode(encoding="utf-8"))
            clientSocket.send(response_headers_raw.encode(encoding="utf-8"))
            # to separate headers from body
            clientSocket.send('\r\n'.encode(encoding="utf-8"))
            clientSocket.send(msg.encode(encoding="utf-8"))
            clientSocket.close()
            continue

        # a status mark of whether the process can run based on the free resources
        waitForRunning = False

        # The processes are running
        numIsRunning = 0

        lockPIDMap.acquire()
        for child in mapPIDtoStatus.copy():
            if mapPIDtoStatus[child] == "running":
                numIsRunning += 1
        if numIsRunning >= numCores:
            waitForRunning = True  # The process need to wait for resources

        # slide windows
        if len(responseMapWindows) >= 100:
            responseMapWindows.pop(0)

        childProcess = os.fork()
        if childProcess != 0:
            responseMapWindows.append([childProcess, [time.time(), -1]])

        if childProcess == 0:
            # This is the child process: run the function and exit
            myFunction(data_, clientSocket)
            os._exit(os.EX_OK)
        else:
            # Append submit time to the responseMapWindows
            if waitForRunning:
                # If there is no free resources (cpu core) for the process to run, then we set the childprocess to sleep.
                mapPIDtoStatus[childProcess] = "waiting"
                os.kill(childProcess, signal.SIGSTOP)

                # Push to priority queue (using burstTime for SRTF logic)
                burstTime = myFunction(data_, clientSocket)
                heapq.heappush(processQueue, (burstTime, childProcess))
            else:
                # If there are free resources (cpu core) for the process to run, then we let the childprocess to run.
                mapPIDtoStatus[childProcess] = "running"
                requestQueue.append(childProcess)

            lockPIDMap.release()
            # The childprocess is running, when it is finished, let the queue find waiting childprocesses
            threadWait = threading.Thread(
                target=waitTermination, args=(childProcess,))
            threadWait.start()


if __name__ == "__main__":
    run()
