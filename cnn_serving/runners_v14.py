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
import heapq

# LSTM-specific imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

current_path = "/app/pythonAction"
TMP_DIR = "/tmp"
BETA = 0.3  # Weight for wait time
processQueue = []


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

processTimestamps = {}  # {pid: (total_wait, last_wait_start_time)}
FUNCTION_HISTORY_KEY = "function_history"
# Menyimpan histori eksekusi proses
processExecutionHistory = {FUNCTION_HISTORY_KEY: []}
processStartTime = {}
processExecutedTime = {}  # {pid: accumulated_executed_seconds}

lockPIDMap = threading.Lock()
requestQueue = []  # queue of child processes
mapPIDtoStatus = {}  # map from pid to status (running, waiting)

processArrivalTimes = {}  # Dictionary to track arrival times of processes
responseMapWindows = []  # map from pid to response

affinity_mask = {0, 1, 2, 3, 4, 5, 6, 7}

# LSTM Model Cache (to avoid retraining on every call)
lstm_model = None
lstm_model_lock = threading.Lock()
lstm_sequence_length = 3  # CHANGED: Reduced from 10 to 3 - works with small datasets
lstm_min_samples = 5      # CHANGED: Added minimum samples before LSTM kicks in

def build_lstm_model(input_shape):
    """
    Build a SMALLER and FASTER LSTM model optimized for small datasets.
    """
    model = Sequential([
        LSTM(8, activation='relu', input_shape=input_shape,  # Reduced from 16 to 8
             return_sequences=False),
        Dropout(0.1),  # Reduced from 0.2 to 0.1
        Dense(4, activation='relu'),  # Reduced from 8 to 4
        Dense(1, activation='relu')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model


def train_lstm_model(history):
    """
    Train LSTM model on execution history.
    OPTIMIZED for small datasets (20-100 samples).
    """
    global lstm_model

    # CHANGED: Use smaller sequence length
    if len(history) < lstm_sequence_length + 1:
        return None  # Not enough data

    try:
        # Normalize the data
        history_np = np.array(history, dtype=np.float32)
        history_min = history_np.min()
        history_max = history_np.max()

        if history_max == history_min:
            return None  # All values are the same, can't train

        history_normalized = (history_np - history_min) / \
            (history_max - history_min)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(history_normalized) - lstm_sequence_length):
            X.append(history_normalized[i:i + lstm_sequence_length])
            y.append(history_normalized[i + lstm_sequence_length])

        if len(X) < 2:  # Need at least 2 sequences
            return None

        X = np.array(X).reshape(-1, lstm_sequence_length, 1)
        y = np.array(y)

        # Train the model (verbose=0 to avoid spam)
        model = build_lstm_model((lstm_sequence_length, 1))
        model.fit(X, y, epochs=5, batch_size=1, verbose=0)  # batch_size=1 for small data

        # Store normalization parameters
        model.history_min = history_min
        model.history_max = history_max

        return model
    except Exception as e:
        print(f"Error training LSTM: {e}")
        return None


def predict_lstm(model, history):
    """
    Use LSTM model to predict the next execution time.
    """
    if model is None or len(history) < lstm_sequence_length:
        return None

    try:
        history_np = np.array(history, dtype=np.float32)

        # Normalize using the model's stored parameters
        history_normalized = (history_np - model.history_min) / \
            (model.history_max - model.history_min)

        # Get the last sequence
        last_sequence = history_normalized[-lstm_sequence_length:].reshape(
            1, lstm_sequence_length, 1)

        # Predict
        prediction_normalized = model.predict(last_sequence, verbose=0)[0][0]

        # Denormalize
        prediction = prediction_normalized * \
            (model.history_max - model.history_min) + model.history_min

        return max(float(prediction), 0)  # Ensure non-negative
    except Exception as e:
        print(f"Error predicting with LSTM: {e}")
        return None


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
    
    # CRITICAL SECTION: Block SIGTSTP during response sending to prevent
    # preemption from interrupting socket writes and causing connection errors
    try:
        # Block SIGTSTP to prevent preemption during response sending
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTSTP})
        
        clientSocket_.send(r.encode(encoding="utf-8"))
        clientSocket_.send(response_headers_raw.encode(encoding="utf-8"))
        # to separate headers from body
        clientSocket_.send('\r\n'.encode(encoding="utf-8"))
        clientSocket_.send(msg.encode(encoding="utf-8"))
    except Exception as e:
        print(f"Error sending response: {e}", flush=True)
    finally:
        # Unblock SIGTSTP after response is sent
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTSTP})
        try:
            clientSocket_.close()
        except:
            pass


# Fungsi EWMA (Exponential Weighted Moving Average)
def calculate_ewma(history, alpha=0.8):
    if not history:
        return 0  # Jika tidak ada data, kembalikan 0
    ewma = history[0]  # Nilai awal
    for val in history[1:]:
        ewma = alpha * val + (1 - alpha) * ewma
    return ewma


def get_burst_time_prediction_lstm():
    """
    LSTM-based prediction optimized for small workloads (20-100 requests).
    Uses aggressive fallback strategy with EWMA.
    """
    global lstm_model
    history = processExecutionHistory[FUNCTION_HISTORY_KEY]

    if not history:
        return 2.0  # Default if no history

    # CHANGED: More aggressive LSTM training (retrain every 5 samples instead of 10)
    if len(history) >= lstm_min_samples:
        with lstm_model_lock:
            # Retrain every 5 samples OR if model doesn't exist
            if lstm_model is None or len(history) % 5 == 0:
                lstm_model = train_lstm_model(history)

        if lstm_model is not None:
            lstm_pred = predict_lstm(lstm_model, history)
            if lstm_pred is not None and lstm_pred > 0:
                # CHANGED: Blend LSTM with EWMA for robustness
                ewma_pred = calculate_ewma(history)
                # 70% LSTM, 30% EWMA for stability
                blended_pred = 0.7 * lstm_pred + 0.3 * ewma_pred
                return blended_pred

    # Fallback to mean/EWMA (before 5 samples or if LSTM fails)
    if len(history) >= 3:
        avg_burst_time = np.mean(history)
        ewma_burst_time = calculate_ewma(history)
        std_dev = np.std(history) if len(history) > 1 else 0

        tsi = (avg_burst_time + ewma_burst_time) / 2
        tsu = max(ALPHA_RT * tsi - BETA_RT * std_dev, 0)
        return tsu
    else:
        # Very early stage: just use mean
        return np.mean(history)
    

# Parameter Mitigasi Ketidakpastian
ALPHA_RT = 0.7  # Faktor koreksi waktu estimasi
BETA_RT = 0.3   # Faktor penalti standar deviasi

# Fungsi Menghitung Remaining Time
def calculate_remaining_time(pid):
    """
    Calculate remaining time using LSTM prediction.
    Uses processExecutedTime to properly track accumulated execution time.
    """
    # Get prediction from LSTM model (centralized prediction logic)
    estimated_burst_time = get_burst_time_prediction_lstm()  # or just reuse the name
    
    # Calculate elapsed time using processExecutedTime (accumulated) + current running segment
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
            individual_wait = acc_wait + (current_time - last_wait if last_wait else 0.0)
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
        total_executed = processExecutedTime.get(childPid, 0.0)
        if childPid in processStartTime:
            total_executed += time.time() - processStartTime[childPid]
        processExecutionHistory[FUNCTION_HISTORY_KEY].append(total_executed)
        
        # Clean up tracking structures for finished process
        processTimestamps.pop(childPid, None)
        processStartTime.pop(childPid, None)
        processExecutedTime.pop(childPid, None)
    except Exception as e:
        print(f"Error removing process {childPid}: {e}")

    # PREEMPTION: Cek apakah ada proses dengan waktu tersisa lebih pendek dari proses yang berjalan
    if processQueue:
        # Clean up stale PIDs from the queue before scheduling
        # A PID is stale if the process no longer exists (not in /proc/{pid})
        def is_process_alive(pid):
            """Check if a process is still alive using /proc filesystem."""
            try:
                # On Linux, /proc/{pid} exists if process is alive
                return os.path.exists(f"/proc/{pid}")
            except Exception:
                return False
        
        # Filter out dead processes from the queue
        original_queue_size = len(processQueue)
        processQueue[:] = [
            (remaining_time, pid) for remaining_time, pid in processQueue
            if pid in mapPIDtoStatus and is_process_alive(pid)
        ]
        
        # Also clean up mapPIDtoStatus for dead processes
        dead_pids = [pid for pid in mapPIDtoStatus if not is_process_alive(pid)]
        for dead_pid in dead_pids:
            mapPIDtoStatus.pop(dead_pid, None)
            processTimestamps.pop(dead_pid, None)
            processStartTime.pop(dead_pid, None)
            processExecutedTime.pop(dead_pid, None)
        
        if original_queue_size != len(processQueue):
            # Rebuild heap after filtering
            try:
                heapq.heapify(processQueue)
            except Exception:
                pass

        # Urutkan queue berdasarkan 1 / remaining_time untuk SRTF
        def priority_selector(process_item):
            _, pid = process_item
            remaining_time = calculate_remaining_time(pid) + 1e-9

            # Calculate individual wait time
            if pid in processTimestamps:
                acc_wait, last_wait = processTimestamps[pid]
                individual_wait_time = acc_wait + (time.time() - last_wait if last_wait else 0.0)
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
                current_remaining = calculate_remaining_time(
                    current_running_pid)
                next_remaining = calculate_remaining_time(nextProcess)

                # PREEMPTION CHECK
                if next_remaining < current_remaining - PREEMPTION_THRESHOLD:
                    print(f"Preempting process {current_running_pid} (remaining: {current_remaining:.2f}s) "
                          f"with process {nextProcess} (remaining: {next_remaining:.2f}s)")

                    # Hentikan proses yang berjalan
                    try:
                        # sebelum SIGTSTP, akumulasikan waktu yang sudah berjalan
                        if current_running_pid in processStartTime:
                            elapsed_since_start = time.time() - processStartTime[current_running_pid]
                            processExecutedTime[current_running_pid] = processExecutedTime.get(current_running_pid, 0) + elapsed_since_start
                            processStartTime.pop(current_running_pid, None)
                        # Use SIGTSTP instead of SIGSTOP so child can block it during response sending
                        os.kill(current_running_pid, signal.SIGTSTP)
                        mapPIDtoStatus[current_running_pid] = "waiting"

                        # set last_wait_start (keep accumulated if any)
                        acc_w, _ = processTimestamps.get(current_running_pid, (0.0, None))
                        processTimestamps[current_running_pid] = (acc_w, time.time())

                        try:
                            heapq.heappush(processQueue, (calculate_remaining_time(current_running_pid), current_running_pid))
                        except Exception as e:
                            print(f"Error pushing process {current_running_pid} back to queue: {e}")
                    except Exception as e:
                        print(
                            f"Error stopping process {current_running_pid}: {e}")

            # Jalankan proses dengan prioritas tertinggi
            removed = False
            for i, item in enumerate(processQueue):
                if item[1] == nextProcess:
                    processQueue.pop(i)
                    # rebuild heap
                    try:
                        heapq.heapify(processQueue)
                    except Exception:
                        pass
                    removed = True
                    break
            if not removed:
                # fallback: leave queue intact (entry might not exist anymore)
                pass
            
            if nextProcess in mapPIDtoStatus:
                mapPIDtoStatus[nextProcess] = "running"

                try:
                    os.kill(nextProcess, signal.SIGCONT)
                    now = time.time()

                    # Before resuming, accumulate wait segment into acc_wait and clear last_wait_start
                    acc_w, last_wait = processTimestamps.get(nextProcess, (0.0, None))
                    if last_wait:
                        acc_w += now - last_wait
                    processTimestamps[nextProcess] = (acc_w, None)

                    # Reset waktu mulai eksekusi
                    processStartTime[nextProcess] = now
                except ProcessLookupError:
                    # This handles the specific race condition where the process is gone
                    print(f"Scheduler: Process {nextProcess} disappeared before it could be resumed.")
                    mapPIDtoStatus.pop(nextProcess, None) # Clean up
                except Exception as e:
                    print(f"Error resuming process {nextProcess}: {e}")
            else:
                # This handles the case where the process was already cleaned up but its PID was still in the queue
                print(f"Scheduler: Stale PID {nextProcess} found in queue, skipping.")

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
        processExecutedTime[blockedID] = processExecutedTime.get(blockedID, 0.0) + elapsed
        processStartTime.pop(blockedID, None)
    for child in mapPIDtoStatus.copy():
        if child in mapPIDtoStatus:
            if mapPIDtoStatus[child] == "waiting":
                mapPIDtoStatus[child] = "running"
                # remove any queued entries for this pid (heapq/queue may contain tuple entries)
                try:
                    # safe linear scan remove (queue is small)
                    for i, item in enumerate(processQueue):
                        if item[1] == child:
                            processQueue.pop(i)
                            heapq.heapify(processQueue)
                            break
                except Exception:
                    pass
                # accumulate wait segment -> clear last_wait, set start time
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
            # write to temp, flush and sync, then atomically move into place
            with open(tmp_proc_path, "wb") as my_blob:
                my_blob.write(blob_val)
                my_blob.flush()
                os.fsync(my_blob.fileno())
            os.replace(tmp_proc_path, proc_path)
        except Exception:
            # best-effort cleanup on failure
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
    numRunning = 0  # number of running processes
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
            heapq.heappush(processQueue, (calculate_remaining_time(blockedID), blockedID))
        except Exception:
            pass
        try:
            os.kill(blockedID, signal.SIGTSTP)
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
    global processStartTime
    global processTimestamps

    try:
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
            response_status_text = 'OK'

            r = '%s %s %s\r\n' % (
                response_proto, response_status, response_status_text)
            try:
                clientSocket.send(r.encode(encoding="utf-8"))
                clientSocket.send(
                    response_headers_raw.encode(encoding="utf-8"))
                clientSocket.send('\r\n'.encode(encoding="utf-8"))
                clientSocket.send(msg.encode(encoding="utf-8"))
                clientSocket.close()
                return
            except:
                clientSocket.close()
                return

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
                # Reset execution history for round separation
                processExecutionHistory[FUNCTION_HISTORY_KEY] = []
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

            response_proto = 'HTTP/1.1'
            response_status = '200'
            response_status_text = 'OK'

            r = '%s %s %s\r\n' % (
                response_proto, response_status, response_status_text)

            clientSocket.send(r.encode(encoding="utf-8"))
            clientSocket.send(response_headers_raw.encode(encoding="utf-8"))
            clientSocket.send('\r\n'.encode(encoding="utf-8"))
            clientSocket.send(msg.encode(encoding="utf-8"))
            clientSocket.close()
            return

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
        if childProcess == 0:
            # Child process: run the function and exit
            myFunction(data_, clientSocket, time.time())
            os._exit(os.EX_OK)
        else:
            # Append submit time to the responseMapWindows
            responseMapWindows.append([childProcess, [time.time(), -1]])
            processStartTime[childProcess] = time.time()

            # store (accumulated_wait_seconds, last_wait_start_timestamp_or_None)
            processTimestamps[childProcess] = (0.0, None)
            
            if waitForRunning:
                # If there is no free resources (cpu core) for the process to run, then we set the childprocess to sleep.
                mapPIDtoStatus[childProcess] = "waiting"
                # Use SIGTSTP instead of SIGSTOP so child can block it during response
                os.kill(childProcess, signal.SIGTSTP)

                # Push to priority queue (using burstTime for SRTF logic)
                estimated_burst_time = calculate_remaining_time(childProcess)
                heapq.heappush(processQueue, (estimated_burst_time, childProcess))

                processStartTime.pop(childProcess, None)
                acc, _ = processTimestamps.get(childProcess, (0.0, None))
                processTimestamps[childProcess] = (acc, time.time())
            else:
                mapPIDtoStatus[childProcess] = "running"
                requestQueue.append(childProcess)

            lockPIDMap.release()
            
            # Monitor child termination in separate thread
            threadWait = threading.Thread(target=waitTermination, args=(childProcess,))
            threadWait.daemon = True
            threadWait.start()
            
            # Parent should NOT close the socket - child process owns it now
            # The child will close it after sending the response in myFunction()

    except Exception as e:
        print(f"Error handling client {address}: {e}", flush=True)
        try:
            clientSocket.close()
        except:
            pass


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
    serverSocket.listen(10)

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
