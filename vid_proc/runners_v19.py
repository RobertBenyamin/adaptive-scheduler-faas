import json
import os
import sys
import signal
import threading
import socket
import numpy as np
import time
import signal
from storage_helper import download_file, upload_file
from river import ensemble
from river import drift
import heapq

current_path = "/app/pythonAction"
TMP_DIR = "/tmp"
BETA = 0.3  # Weight for wait time
processQueue = []


def signal_handler(sig, frame):
    serverSocket_.close()
    sys.exit(0)

# 19v1
class SA_RF_CDD_Wrapper:
    def __init__(self):
        # Using AdaptiveRandomForestRegressor from River (river 0.14.0 API)
        # Implements Hoeffding Trees + ADWIN (Drift Detection) internally
        self.model = ensemble.AdaptiveRandomForestRegressor(
            n_models=10,      # Number of trees (N)
            seed=42,
            # Equivalent to nmin before split (Hoeffding Bound)
            grace_period=50,
            drift_detector=drift.ADWIN(delta=0.002)  # Concept Drift Detector
        )
        self.last_arrival_time = 0

    def extract_features(self, history, current_arrival_time, last_arrival):
        """
        Transform raw history into features
        Features: Lags, Window Stats, Volatility, Delta, Inter-Arrival
        """
        if len(history) < 10:
            # Cold start handling: return default safe features
            return {
                "lag_1": 0, "lag_2": 0, "lag_3": 0,
                "mean_5": 0, "std_10": 0,
                "delta_1_2": 0,
                "inter_arrival": 0
            }

        # 1. Lag Features (Autocorrelation)
        lag_1 = history[-1]
        lag_2 = history[-2]
        lag_3 = history[-3]

        # 2. Window Statistics (Short-term Trend)
        mean_5 = np.mean(history[-5:])

        # 3. Volatility Measures (Variability)
        std_10 = np.std(history[-10:])

        # 4. Delta Features (Acceleration)
        delta_1_2 = lag_1 - lag_2

        # 5. Inter-Arrival Time (Contention Indicator)
        inter_arrival = current_arrival_time - last_arrival if last_arrival > 0 else 0

        return {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "mean_5": mean_5,
            "std_10": std_10,
            "delta_1_2": delta_1_2,
            "inter_arrival": inter_arrival
        }

    def predict(self, history):
        """
        Predict burst time using stream-based model (very fast, O(depth))
        """
        features = self.extract_features(
            history, time.time(), self.last_arrival_time)
        prediction = self.model.predict_one(features)
        # Return default if model hasn't learned yet
        return prediction if prediction is not None else 2.0

    def learn(self, history, arrival_time, actual_duration):
        """
        Update model incrementally (O(1))
        """
        features = self.extract_features(
            history, arrival_time, self.last_arrival_time)
        self.model.learn_one(features, actual_duration)
        self.last_arrival_time = arrival_time


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

processTimestamps = {}  # {pid: (total_wait, last_wait_start_time)}
FUNCTION_HISTORY_KEY = "function_history"
# Stores process execution history
processExecutionHistory = {FUNCTION_HISTORY_KEY: []}
processStartTime = {}
processExecutedTime = {}  # {pid: accumulated_executed_seconds}
processArrivalTime = {}  # {pid: arrival_time} - for learning

lockPIDMap = threading.Lock()
requestQueue = []  # queue of child processes
mapPIDtoStatus = {}  # map from pid to status (running, waiting)

processArrivalTimes = {}  # Dictionary to track arrival times of processes
responseMapWindows = []  # map from pid to response

affinity_mask = {0, 1, 2, 3, 4, 5, 6, 7}

# SA-RF-CDD model instance (replaces sklearn RandomForest)
sa_rf_cdd_model = SA_RF_CDD_Wrapper()
model_lock = threading.Lock()


# The function to update the core nums by request.
def updateThread():
    # Shared variable: numCores
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


def myFunction(data_, clientSocket_, arrival_time):
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

        # Calculate turnaround time and add it to the response
        turnaround_time = time.time() - arrival_time
        result["turnaround_time"] = turnaround_time

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


def get_burst_time_prediction():
    """
    Get burst time prediction using SA-RF-CDD (stream-based, no caching needed).
    The model is already incremental, so prediction is O(depth).
    """
    history = processExecutionHistory[FUNCTION_HISTORY_KEY]

    with model_lock:
        prediction = sa_rf_cdd_model.predict(history)

    return prediction if prediction is not None else 2.0


# Function to calculate Remaining Time
def calculate_remaining_time(pid):
    """
    Calculate remaining time using SA-RF-CDD prediction.
    """
    # Get prediction from stream-based model
    estimated_burst_time = get_burst_time_prediction()

    # Calculate elapsed time
    elapsed_time = processExecutedTime.get(pid, 0)
    if pid in processStartTime:
        elapsed_time += time.time() - processStartTime[pid]

    # Remaining = total - elapsed
    remaining_time = max(estimated_burst_time - elapsed_time, 0)

    return remaining_time


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
            acc_wait, last_wait = processTimestamps[pid]
            individual_wait = acc_wait + \
                (current_time - last_wait if last_wait else 0.0)
            total_wait_time += individual_wait

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


# Maximum time threshold before preemption occurs (in seconds)
PREEMPTION_THRESHOLD = 4


def waitTermination(childPid):
    """
    Wait for process to finish or replace it if there's a higher priority process with preemption.
    """
    global processQueue, mapPIDtoStatus

    os.waitpid(childPid, 0)  # Wait until process finishes

    lockPIDMap.acquire()

    try:
        # Remove process from status map
        mapPIDtoStatus.pop(childPid, None)

        # Calculate and save burst time to history
        total_executed = processExecutedTime.get(childPid, 0.0)
        if childPid in processStartTime:
            total_executed += time.time() - processStartTime[childPid]

        # Get arrival time for learning
        arrival_time = processArrivalTime.get(childPid, time.time())

        # Learn from this execution (incremental update)
        history_context = list(processExecutionHistory[FUNCTION_HISTORY_KEY])
        with model_lock:
            sa_rf_cdd_model.learn(
                history_context, arrival_time, total_executed)

        # Add to history after learning
        processExecutionHistory[FUNCTION_HISTORY_KEY].append(total_executed)

        # Limit history size to prevent memory leak
        if len(processExecutionHistory[FUNCTION_HISTORY_KEY]) > 1000:
            processExecutionHistory[FUNCTION_HISTORY_KEY].pop(0)

        # Clean up tracking structures for finished process
        processTimestamps.pop(childPid, None)
        processStartTime.pop(childPid, None)
        processExecutedTime.pop(childPid, None)
        processArrivalTime.pop(childPid, None)
    except Exception as e:
        print(f"Error removing process {childPid}: {e}")

    # PREEMPTION: Check if there's a process with shorter remaining time than running process
    if processQueue:
        # Sort queue based on 1 / remaining_time for SRTF
        def priority_selector(process_item):
            _, pid = process_item
            remaining_time = calculate_remaining_time(pid) + 1e-9

            # Calculate individual wait time
            if pid in processTimestamps:
                acc_wait, last_wait = processTimestamps[pid]
                individual_wait_time = acc_wait + \
                    (time.time() - last_wait if last_wait else 0.0)
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

        # Get process with highest priority
        next_process_candidates = sorted(
            processQueue, key=priority_selector, reverse=True)

        if next_process_candidates:
            _, nextProcess = next_process_candidates[0]
            current_running_pid = None

            # Find currently running process
            for pid, status in mapPIDtoStatus.items():
                if status == "running":
                    current_running_pid = pid
                    break

            # If there's a running process, check if it should be preempted
            if current_running_pid:
                current_remaining = calculate_remaining_time(
                    current_running_pid)
                next_remaining = calculate_remaining_time(nextProcess)

                # PREEMPTION CHECK
                if next_remaining < current_remaining - PREEMPTION_THRESHOLD:
                    print(f"Preempting process {current_running_pid} (remaining: {current_remaining:.2f}s) "
                          f"with process {nextProcess} (remaining: {next_remaining:.2f}s)")

                    # Stop running process
                    try:
                        # Before SIGSTOP, accumulate elapsed time
                        if current_running_pid in processStartTime:
                            elapsed_since_start = time.time(
                            ) - processStartTime[current_running_pid]
                            processExecutedTime[current_running_pid] = processExecutedTime.get(
                                current_running_pid, 0) + elapsed_since_start
                            processStartTime.pop(current_running_pid, None)
                        os.kill(current_running_pid, signal.SIGSTOP)
                        mapPIDtoStatus[current_running_pid] = "waiting"

                        # Set last_wait_start (keep accumulated if any)
                        acc_w, _ = processTimestamps.get(
                            current_running_pid, (0.0, None))
                        processTimestamps[current_running_pid] = (
                            acc_w, time.time())

                        try:
                            heapq.heappush(processQueue, (calculate_remaining_time(
                                current_running_pid), current_running_pid))
                        except Exception as e:
                            print(
                                f"Error pushing process {current_running_pid} back to queue: {e}")
                    except Exception as e:
                        print(
                            f"Error stopping process {current_running_pid}: {e}")

            # Run process with highest priority
            removed = False
            for i, item in enumerate(processQueue):
                if item[1] == nextProcess:
                    processQueue.pop(i)
                    # Rebuild heap
                    try:
                        heapq.heapify(processQueue)
                    except Exception:
                        pass
                    removed = True
                    break
            if not removed:
                # Fallback: leave queue intact (entry might not exist anymore)
                pass

            if nextProcess in mapPIDtoStatus:
                mapPIDtoStatus[nextProcess] = "running"

                try:
                    os.kill(nextProcess, signal.SIGCONT)
                    now = time.time()

                    # Before resuming, accumulate wait segment into acc_wait and clear last_wait_start
                    acc_w, last_wait = processTimestamps.get(
                        nextProcess, (0.0, None))
                    if last_wait:
                        acc_w += now - last_wait
                    processTimestamps[nextProcess] = (acc_w, None)

                    # Reset execution start time
                    processStartTime[nextProcess] = now
                except ProcessLookupError:
                    # This handles the specific race condition where the process is gone
                    print(
                        f"Scheduler: Process {nextProcess} disappeared before it could be resumed.")
                    mapPIDtoStatus.pop(nextProcess, None)  # Clean up
                except Exception as e:
                    print(f"Error resuming process {nextProcess}: {e}")
            else:
                # This handles the case where the process was already cleaned up but its PID was still in the queue
                print(
                    f"Scheduler: Stale PID {nextProcess} found in queue, skipping.")

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

    my_id = blockedID

    lockPIDMap.acquire()
    mapPIDtoStatus[blockedID] = "blocked"
    # Mark blocked: accumulate running time before blocking so accounting stays correct
    # (do this while holding lock)
    if blockedID in processStartTime:
        elapsed = time.time() - processStartTime[blockedID]
        processExecutedTime[blockedID] = processExecutedTime.get(
            blockedID, 0.0) + elapsed
        processStartTime.pop(blockedID, None)
    for child in mapPIDtoStatus.copy():
        if child in mapPIDtoStatus:
            if mapPIDtoStatus[child] == "waiting":
                mapPIDtoStatus[child] = "running"
                # Remove any queued entries for this pid (heapq/queue may contain tuple entries)
                try:
                    # Safe linear scan remove (queue is small)
                    for i, item in enumerate(processQueue):
                        if item[1] == child:
                            processQueue.pop(i)
                            heapq.heapify(processQueue)
                            break
                except Exception:
                    pass
                # Accumulate wait segment -> clear last_wait, set start time
                now = time.time()
                acc_w, last_wait = processTimestamps.get(child, (0.0, None))
                if last_wait:
                    acc_w += now - last_wait
                processTimestamps[child] = (acc_w, None)
                processStartTime[child] = now
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

            download_file(blobName, os.path.join(TMP_DIR, blobName))
            with open(os.path.join(TMP_DIR, blobName), "rb") as file:
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
        proc_path = os.path.join(TMP_DIR, proc_blob_name)
        tmp_proc_path = proc_path + ".part"
        try:
            # Write to temp, flush and sync, then atomically move into place
            with open(tmp_proc_path, "wb") as my_blob:
                my_blob.write(blob_val)
                my_blob.flush()
                os.fsync(my_blob.fileno())
            os.replace(tmp_proc_path, proc_path)
        except Exception:
            # Best-effort cleanup on failure
            try:
                if os.path.exists(tmp_proc_path):
                    os.remove(tmp_proc_path)
            except:
                pass
    else:
        fReadname = message["value"]
        fRead = open(fReadname, "rb")
        value = fRead.read()

        upload_file(f"{current_path}/{value}", f"files/{blobName}")
        blob_val = "none"

    lockPIDMap.acquire()
    numRunning = 0  # Number of running processes
    for child in mapPIDtoStatus.copy():
        if mapPIDtoStatus[child] == "running":
            numRunning += 1
    if numRunning < numCores:
        mapPIDtoStatus[blockedID] = "running"
        now = time.time()
        acc_w, last_wait = processTimestamps.get(blockedID, (0.0, None))
        if last_wait:
            acc_w += now - last_wait
        processTimestamps[blockedID] = (acc_w, None)
        processStartTime[blockedID] = now
        try:
            os.kill(blockedID, signal.SIGCONT)
        except:
            pass
    else:
        mapPIDtoStatus[blockedID] = "waiting"
        acc_w, _ = processTimestamps.get(blockedID, (0.0, None))
        processTimestamps[blockedID] = (acc_w, time.time())
        try:
            heapq.heappush(
                processQueue, (calculate_remaining_time(blockedID), blockedID))
        except Exception:
            pass
        try:
            os.kill(blockedID, signal.SIGSTOP)
        except:
            pass
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


def handle_client_connection(clientSocket, address):
    """Handle a single client connection in a separate thread"""
    global requestQueue
    global mapPIDtoStatus
    global numCores
    global responseMapWindows
    global affinity_mask
    global processQueue

    try:
        print("Accept a new connection from %s" % str(address), flush=True)

        data_ = b''
        data_ += clientSocket.recv(1024)
        dataStr = data_.decode('UTF-8')

        # Check for Host header
        if 'Host' not in dataStr:
            msg = 'OK'
            response_headers = {
                'Content-Type': 'text/html; encoding=utf8',
                'Content-Length': len(msg),
                'Connection': 'close',
            }
            response_headers_raw = ''.join('%s: %s\r\n' % (
                k, v) for k, v in response_headers.items())
            r = 'HTTP/1.1 200 OK\r\n'
            try:
                clientSocket.send(r.encode(encoding="utf-8"))
                clientSocket.send(
                    response_headers_raw.encode(encoding="utf-8"))
                clientSocket.send('\r\n'.encode(encoding="utf-8"))
                clientSocket.send(msg.encode(encoding="utf-8"))
            except:
                pass
            finally:
                clientSocket.close()
            return

        # Parse message
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

        # Handle numCores, Q, Clear messages
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
            elif "Q" in message:
                i = []
                for responseTime in responseMapWindows:
                    if responseTime[1][1] != -1:
                        i.append(responseTime[1][1] - responseTime[1][0])
                result = {"p95": np.percentile(i, 95) if len(i) > 0 else 0}
                result["affinity_mask"] = list(affinity_mask)
                result["numCores"] = numCores
                msg = json.dumps(result)
                responseFlag = True
            elif "Clear" in message:
                responseMapWindows = []
                # Reset execution history for round separation
                processExecutionHistory[FUNCTION_HISTORY_KEY] = []
                # Reset SA-RF-CDD model to fresh state
                with model_lock:
                    global sa_rf_cdd_model
                    sa_rf_cdd_model = SA_RF_CDD_Wrapper()
                result = {"Response": "History Reset"}
                msg = json.dumps(result)
                responseFlag = True

        if responseFlag:
            response_headers = {
                'Content-Type': 'text/html; encoding=utf8',
                'Content-Length': len(msg),
                'Connection': 'close',
            }
            response_headers_raw = ''.join('%s: %s\r\n' % (
                k, v) for k, v in response_headers.items())
            r = 'HTTP/1.1 200 OK\r\n'
            try:
                clientSocket.send(r.encode(encoding="utf-8"))
                clientSocket.send(
                    response_headers_raw.encode(encoding="utf-8"))
                clientSocket.send('\r\n'.encode(encoding="utf-8"))
                clientSocket.send(msg.encode(encoding="utf-8"))
            except:
                pass
            finally:
                clientSocket.close()
            return

        # Record arrival time for this request
        arrival_time = time.time()

        # Get burst time prediction using SA-RF-CDD
        estimated_burst_time = get_burst_time_prediction()

        # A status mark of whether the process can run based on the free resources
        waitForRunning = False

        # The processes are running
        numIsRunning = 0

        lockPIDMap.acquire()
        for child in mapPIDtoStatus.copy():
            if mapPIDtoStatus[child] == "running":
                numIsRunning += 1
        if numIsRunning >= numCores:
            waitForRunning = True  # The process need to wait for resources

        # Slide windows
        if len(responseMapWindows) >= 100:
            responseMapWindows.pop(0)

        childProcess = os.fork()
        if childProcess == 0:
            # Child process: run the function and exit
            myFunction(data_, clientSocket, arrival_time)
            os._exit(os.EX_OK)
        else:
            # Append submit time to the responseMapWindows
            responseMapWindows.append([childProcess, [time.time(), -1]])
            processStartTime[childProcess] = time.time()
            # Store arrival time for learning
            processArrivalTime[childProcess] = arrival_time

            # Store (accumulated_wait_seconds, last_wait_start_timestamp_or_None)
            processTimestamps[childProcess] = (0.0, None)

            if waitForRunning:
                # If there is no free resources (cpu core) for the process to run, then we set the childprocess to sleep.
                mapPIDtoStatus[childProcess] = "waiting"
                os.kill(childProcess, signal.SIGSTOP)

                # Push to priority queue (using burstTime for SRTF logic)
                heapq.heappush(
                    processQueue, (estimated_burst_time, childProcess))

                processStartTime.pop(childProcess, None)
                acc, _ = processTimestamps.get(childProcess, (0.0, None))
                processTimestamps[childProcess] = (acc, time.time())
            else:
                mapPIDtoStatus[childProcess] = "running"
                requestQueue.append(childProcess)

            lockPIDMap.release()

            # Monitor child termination in separate thread
            threadWait = threading.Thread(
                target=waitTermination, args=(childProcess,))
            threadWait.daemon = True
            threadWait.start()

    except Exception as e:
        print(f"Error handling client {address}: {e}", flush=True)
    finally:
        try:
            clientSocket.close()
        except:
            pass


def run():
    global serverSocket_
    global actionModule
    global numCores

    numCores = 8
    os.sched_setaffinity(0, affinity_mask)
    print("Welcome... ", numCores)

    myHost = '0.0.0.0'
    myPort = int(os.environ.get('PORT', 8081))

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind((myHost, myPort))
    serverSocket.listen(10)

    serverSocket_ = serverSocket

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
    threadUpdate.daemon = True
    threadUpdate.start()

    # Monitor I/O Block
    threadIntercept = threading.Thread(target=IOThread)
    threadIntercept.daemon = True
    threadIntercept.start()

    # Accept connections and handle each in a separate thread
    while True:
        try:
            (clientSocket, address) = serverSocket.accept()
            # Handle each connection in a separate thread
            handler_thread = threading.Thread(
                target=handle_client_connection,
                args=(clientSocket, address)
            )
            handler_thread.daemon = True
            handler_thread.start()
        except Exception as e:
            print(f"Error accepting connection: {e}", flush=True)


if __name__ == "__main__":
    run()
