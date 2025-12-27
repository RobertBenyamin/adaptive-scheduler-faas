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
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports for Online LSTM
import tensorflow as tf
from collections import deque

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

# ============================================================================
# ONLINE LSTM WITH ONLINE GRADIENT DESCENT (OGD) IMPLEMENTATION
# ============================================================================

class OnlineLSTMModel:
    """
    Online LSTM that updates weights incrementally using Online Gradient Descent.
    Each new observation triggers a single gradient update step.
    """
    
    def __init__(self, sequence_length=3, hidden_units=8, learning_rate=0.01):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        # Sequence buffer to store recent observations
        self.sequence_buffer = deque(maxlen=sequence_length + 1)
        
        # Running statistics for online normalization
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_samples = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # Build the LSTM model
        self.model = self._build_model()
        
        # Optimizer for online gradient descent
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Loss function
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Track training metrics
        self.total_updates = 0
        self.cumulative_loss = 0.0
        
        # Learning rate decay parameters
        self.decay_rate = 0.999
        self.min_learning_rate = 0.001
        
    def _build_model(self):
        """Build a lightweight LSTM model for online learning."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                self.hidden_units, 
                activation='tanh',
                recurrent_activation='sigmoid',
                input_shape=(self.sequence_length, 1),
                return_sequences=False,
                # Use smaller recurrent dropout for stability
                recurrent_dropout=0.0,
                dropout=0.0
            ),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')  # Linear for regression
        ])
        return model
    
    def _update_running_stats(self, value):
        """Update running statistics for online normalization (Welford's algorithm)."""
        self.n_samples += 1
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Welford's online algorithm for mean and variance
        delta = value - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = value - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.n_samples
        
    def _normalize(self, value):
        """Normalize a value using running statistics."""
        if self.max_val == self.min_val:
            return 0.0
        return (value - self.min_val) / (self.max_val - self.min_val + 1e-8)
    
    def _denormalize(self, value):
        """Denormalize a value back to original scale."""
        if self.max_val == self.min_val:
            return self.running_mean
        return value * (self.max_val - self.min_val) + self.min_val
    
    def _decay_learning_rate(self):
        """Apply learning rate decay for stability."""
        self.learning_rate = max(
            self.learning_rate * self.decay_rate,
            self.min_learning_rate
        )
        self.optimizer.learning_rate.assign(self.learning_rate)
    
    @tf.function
    def _train_step(self, X, y):
        """
        Perform a single gradient descent step (Online Gradient Descent).
        This is the core of online learning.
        """
        with tf.GradientTape() as tape:
            # Forward pass
            prediction = self.model(X, training=True)
            # Compute loss
            loss = self.loss_fn(y, prediction)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        clipped_gradients = [
            tf.clip_by_value(g, -1.0, 1.0) if g is not None else g 
            for g in gradients
        ]
        
        # Apply gradients (single update step)
        self.optimizer.apply_gradients(
            zip(clipped_gradients, self.model.trainable_variables)
        )
        
        return loss, prediction
    
    def partial_fit(self, new_value):
        """
        Online learning: update model with a single new observation.
        This implements Online Gradient Descent for LSTM.
        
        Args:
            new_value: The new burst time observation
            
        Returns:
            loss: The loss for this update (None if not enough data)
        """
        # Update running statistics
        self._update_running_stats(new_value)
        
        # Normalize and add to buffer
        normalized_value = self._normalize(new_value)
        self.sequence_buffer.append(normalized_value)
        
        # Need at least sequence_length + 1 samples to train
        if len(self.sequence_buffer) < self.sequence_length + 1:
            return None
        
        # Prepare training data from buffer
        # X: sequence of length `sequence_length`
        # y: the next value (target)
        buffer_list = list(self.sequence_buffer)
        X_seq = np.array(buffer_list[:-1], dtype=np.float32)
        y_target = np.array([buffer_list[-1]], dtype=np.float32)
        
        # Reshape for LSTM: (batch_size=1, sequence_length, features=1)
        X_seq = X_seq.reshape(1, self.sequence_length, 1)
        y_target = y_target.reshape(1, 1)
        
        # Convert to tensors
        X_tensor = tf.convert_to_tensor(X_seq, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_target, dtype=tf.float32)
        
        # Perform single gradient update (Online Gradient Descent)
        loss, _ = self._train_step(X_tensor, y_tensor)
        
        # Update metrics
        self.total_updates += 1
        self.cumulative_loss += float(loss)
        
        # Decay learning rate for stability
        self._decay_learning_rate()
        
        return float(loss)
    
    def predict(self):
        """
        Predict the next burst time using the current model state.
        
        Returns:
            Predicted burst time (denormalized) or None if not enough data
        """
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        try:
            # Get the last `sequence_length` values
            buffer_list = list(self.sequence_buffer)
            X_seq = np.array(buffer_list[-self.sequence_length:], dtype=np.float32)
            X_seq = X_seq.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            X_tensor = tf.convert_to_tensor(X_seq, dtype=tf.float32)
            prediction_normalized = self.model(X_tensor, training=False).numpy()[0][0]
            
            # Denormalize
            prediction = self._denormalize(prediction_normalized)
            
            # Ensure non-negative
            return max(float(prediction), 0.0)
        
        except Exception as e:
            print(f"Online LSTM prediction error: {e}")
            return None
    
    def get_average_loss(self):
        """Get the average loss across all updates."""
        if self.total_updates == 0:
            return 0.0
        return self.cumulative_loss / self.total_updates
    
    def reset(self):
        """Reset the model for a new round/experiment."""
        self.sequence_buffer.clear()
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_samples = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.total_updates = 0
        self.cumulative_loss = 0.0
        self.learning_rate = self.initial_learning_rate
        self.optimizer.learning_rate.assign(self.learning_rate)
        # Rebuild model to reset weights
        self.model = self._build_model()


# Global Online LSTM instance
online_lstm_model = None
online_lstm_lock = threading.Lock()

# Configuration
ONLINE_LSTM_SEQUENCE_LENGTH = 3
ONLINE_LSTM_HIDDEN_UNITS = 8
ONLINE_LSTM_LEARNING_RATE = 0.01
ONLINE_LSTM_MIN_SAMPLES = 3  # Minimum samples before using LSTM predictions


def initialize_online_lstm():
    """Initialize the Online LSTM model."""
    global online_lstm_model
    with online_lstm_lock:
        if online_lstm_model is None:
            online_lstm_model = OnlineLSTMModel(
                sequence_length=ONLINE_LSTM_SEQUENCE_LENGTH,
                hidden_units=ONLINE_LSTM_HIDDEN_UNITS,
                learning_rate=ONLINE_LSTM_LEARNING_RATE
            )
            print(f"Online LSTM initialized: seq_len={ONLINE_LSTM_SEQUENCE_LENGTH}, "
                  f"hidden={ONLINE_LSTM_HIDDEN_UNITS}, lr={ONLINE_LSTM_LEARNING_RATE}")


def online_lstm_update(burst_time):
    """
    Update the Online LSTM with a new burst time observation.
    Called when a process completes.
    
    Args:
        burst_time: The actual execution time of the completed process
    """
    global online_lstm_model
    
    initialize_online_lstm()
    
    with online_lstm_lock:
        try:
            loss = online_lstm_model.partial_fit(burst_time)
            if loss is not None:
                print(f"Online LSTM updated: burst_time={burst_time:.4f}, "
                      f"loss={loss:.6f}, total_updates={online_lstm_model.total_updates}")
        except Exception as e:
            print(f"Online LSTM update error: {e}")


def get_burst_time_prediction_online_lstm():
    """
    Get burst time prediction using Online LSTM with OGD.
    Falls back to EWMA if not enough samples.
    
    Returns:
        Predicted burst time
    """
    global online_lstm_model
    
    initialize_online_lstm()
    
    history = processExecutionHistory[FUNCTION_HISTORY_KEY]
    
    if not history:
        return 2.0  # Default if no history
    
    with online_lstm_lock:
        # Try LSTM prediction if we have enough samples
        if online_lstm_model.n_samples >= ONLINE_LSTM_MIN_SAMPLES:
            lstm_pred = online_lstm_model.predict()
            
            if lstm_pred is not None and lstm_pred > 0:
                # Blend with EWMA for robustness (70% LSTM, 30% EWMA)
                ewma_pred = calculate_ewma(history)
                blended_pred = 0.7 * lstm_pred + 0.3 * ewma_pred
                return blended_pred
    
    # Fallback to EWMA/mean
    if len(history) >= 3:
        avg_burst_time = np.mean(history)
        ewma_burst_time = calculate_ewma(history)
        std_dev = np.std(history) if len(history) > 1 else 0
        
        tsi = (avg_burst_time + ewma_burst_time) / 2
        tsu = max(ALPHA_RT * tsi - BETA_RT * std_dev, 0)
        return tsu
    else:
        return np.mean(history) if history else 2.0


def reset_online_lstm():
    """Reset the Online LSTM model (for new rounds/experiments)."""
    global online_lstm_model
    with online_lstm_lock:
        if online_lstm_model is not None:
            online_lstm_model.reset()
            print("Online LSTM model reset")


# ============================================================================
# END OF ONLINE LSTM IMPLEMENTATION
# ============================================================================


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


# Parameter Mitigasi Ketidakpastian
ALPHA_RT = 0.7  # Faktor koreksi waktu estimasi
BETA_RT = 0.3   # Faktor penalti standar deviasi


# Fungsi Menghitung Remaining Time - NOW USES ONLINE LSTM
def calculate_remaining_time(pid):
    """
    Calculate remaining time using Online LSTM prediction with OGD.
    Uses processExecutedTime to properly track accumulated execution time.
    """
    # Get prediction from Online LSTM model
    estimated_burst_time = get_burst_time_prediction_online_lstm()
    
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
    NOW INCLUDES ONLINE LSTM UPDATE when process completes.
    """
    global processQueue, mapPIDtoStatus

    os.waitpid(childPid, 0)  # Tunggu hingga proses selesai

    lockPIDMap.acquire()

    try:
        # Hapus proses dari status map
        mapPIDtoStatus.pop(childPid, None)

        # Calculate total burst time for this process
        total_executed = processExecutedTime.get(childPid, 0.0)
        if childPid in processStartTime:
            total_executed += time.time() - processStartTime[childPid]
        
        # Store burst time to history (for fallback/EWMA)
        processExecutionHistory[FUNCTION_HISTORY_KEY].append(total_executed)
        
        # ============================================================
        # ONLINE LSTM UPDATE: Update model with completed process's burst time
        # This is the key Online Gradient Descent step!
        # ============================================================
        online_lstm_update(total_executed)
        # ============================================================
        
        # Clean up tracking structures for finished process
        processTimestamps.pop(childPid, None)
        processStartTime.pop(childPid, None)
        processExecutedTime.pop(childPid, None)
    except Exception as e:
        print(f"Error removing process {childPid}: {e}")

    # PREEMPTION: Cek apakah ada proses dengan waktu tersisa lebih pendek dari proses yang berjalan
    if processQueue:
        # Clean up stale PIDs from the queue before scheduling
        def is_process_alive(pid):
            """Check if a process is still alive using /proc filesystem."""
            try:
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
            try:
                heapq.heapify(processQueue)
            except Exception:
                pass

        # Urutkan queue berdasarkan priority untuk SRTF
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

                    try:
                        if current_running_pid in processStartTime:
                            elapsed_since_start = time.time() - processStartTime[current_running_pid]
                            processExecutedTime[current_running_pid] = processExecutedTime.get(current_running_pid, 0) + elapsed_since_start
                            processStartTime.pop(current_running_pid, None)
                        os.kill(current_running_pid, signal.SIGTSTP)
                        mapPIDtoStatus[current_running_pid] = "waiting"

                        acc_w, _ = processTimestamps.get(current_running_pid, (0.0, None))
                        processTimestamps[current_running_pid] = (acc_w, time.time())

                        try:
                            heapq.heappush(processQueue, (calculate_remaining_time(current_running_pid), current_running_pid))
                        except Exception as e:
                            print(f"Error pushing process {current_running_pid} back to queue: {e}")
                    except Exception as e:
                        print(f"Error stopping process {current_running_pid}: {e}")

            # Jalankan proses dengan prioritas tertinggi
            removed = False
            for i, item in enumerate(processQueue):
                if item[1] == nextProcess:
                    processQueue.pop(i)
                    try:
                        heapq.heapify(processQueue)
                    except Exception:
                        pass
                    removed = True
                    break
            
            if nextProcess in mapPIDtoStatus:
                mapPIDtoStatus[nextProcess] = "running"

                try:
                    os.kill(nextProcess, signal.SIGCONT)
                    now = time.time()

                    acc_w, last_wait = processTimestamps.get(nextProcess, (0.0, None))
                    if last_wait:
                        acc_w += now - last_wait
                    processTimestamps[nextProcess] = (acc_w, None)

                    processStartTime[nextProcess] = now
                except ProcessLookupError:
                    print(f"Scheduler: Process {nextProcess} disappeared before it could be resumed.")
                    mapPIDtoStatus.pop(nextProcess, None)
                except Exception as e:
                    print(f"Error resuming process {nextProcess}: {e}")
            else:
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
    if blockedID in processStartTime:
        elapsed = time.time() - processStartTime[blockedID]
        processExecutedTime[blockedID] = processExecutedTime.get(blockedID, 0.0) + elapsed
        processStartTime.pop(blockedID, None)
    for child in mapPIDtoStatus.copy():
        if child in mapPIDtoStatus:
            if mapPIDtoStatus[child] == "waiting":
                mapPIDtoStatus[child] = "running"
                try:
                    for i, item in enumerate(processQueue):
                        if item[1] == child:
                            processQueue.pop(i)
                            heapq.heapify(processQueue)
                            break
                except Exception:
                    pass
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
            with open(tmp_proc_path, "wb") as my_blob:
                my_blob.write(blob_val)
                my_blob.flush()
                os.fsync(my_blob.fileno())
            os.replace(tmp_proc_path, proc_path)
        except Exception:
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
    numRunning = 0
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
                
                # Add Online LSTM stats to response
                with online_lstm_lock:
                    if online_lstm_model is not None:
                        result["lstm_updates"] = online_lstm_model.total_updates
                        result["lstm_avg_loss"] = online_lstm_model.get_average_loss()
                        result["lstm_learning_rate"] = online_lstm_model.learning_rate
                
                msg = json.dumps(result)
                responseFlag = True

            if "Clear" in message:
                responseMapWindows = []
                # Reset execution history for round separation
                processExecutionHistory[FUNCTION_HISTORY_KEY] = []
                
                # Reset Online LSTM model for new round
                reset_online_lstm()
                
                result = {"Response": "History and Online LSTM Reset"}
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
                mapPIDtoStatus[childProcess] = "waiting"
                os.kill(childProcess, signal.SIGTSTP)

                # Push to priority queue using Online LSTM prediction
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

    except Exception as e:
        print(f"Error handling client {address}: {e}", flush=True)
        try:
            clientSocket.close()
        except:
            pass


def run():
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
    
    # Initialize Online LSTM at startup
    initialize_online_lstm()
    print("Online LSTM with Online Gradient Descent initialized")

    # Set the address and port
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