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
from collections import deque
warnings.filterwarnings('ignore')

# CRITICAL: Set TensorFlow to not allocate GPU memory and reduce threads
# This MUST be done before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

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
# ONLINE LSTM WITH ONLINE GRADIENT DESCENT (OGD) - FORK-SAFE IMPLEMENTATION
# Uses NumPy-based LSTM to avoid TensorFlow fork issues
# ============================================================================

class OnlineLSTMNumpy:
    """
    Online LSTM implemented in pure NumPy for fork-safety.
    Updates weights incrementally using Online Gradient Descent.
    
    This avoids TensorFlow's fork-unsafety issues completely.
    """
    
    def __init__(self, sequence_length=3, hidden_size=8, learning_rate=0.01):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.input_size = 1
        self.output_size = 1
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        # Initialize LSTM weights using Xavier initialization
        self._init_weights()
        
        # Sequence buffer for online learning
        self.sequence_buffer = deque(maxlen=sequence_length + 1)
        
        # Running statistics for online normalization (Welford's algorithm)
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_samples = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # Hidden and cell states (maintained across predictions for true online learning)
        self.h = np.zeros((1, hidden_size))
        self.c = np.zeros((1, hidden_size))
        
        # Training metrics
        self.total_updates = 0
        self.cumulative_loss = 0.0
        
        # Learning rate decay
        self.decay_rate = 0.999
        self.min_learning_rate = 0.001
        
        # Gradient clipping threshold
        self.clip_value = 1.0
        
        # Adam optimizer states
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step for Adam
        
    def _init_weights(self):
        """Initialize LSTM weights using Xavier initialization."""
        h = self.hidden_size
        i = self.input_size
        
        # Xavier scale factors
        scale_ih = np.sqrt(2.0 / (i + h))
        scale_hh = np.sqrt(2.0 / (h + h))
        scale_ho = np.sqrt(2.0 / (h + self.output_size))
        
        # LSTM weights: [input_gate, forget_gate, cell_gate, output_gate]
        # Input to hidden weights (4 gates)
        self.W_ih = np.random.randn(i, 4 * h).astype(np.float32) * scale_ih
        # Hidden to hidden weights (4 gates)
        self.W_hh = np.random.randn(h, 4 * h).astype(np.float32) * scale_hh
        # Biases for gates
        self.b_ih = np.zeros((1, 4 * h), dtype=np.float32)
        self.b_hh = np.zeros((1, 4 * h), dtype=np.float32)
        
        # Set forget gate bias to 1.0 for better gradient flow
        self.b_ih[0, h:2*h] = 1.0
        self.b_hh[0, h:2*h] = 1.0
        
        # Output layer weights
        self.W_out = np.random.randn(h, self.output_size).astype(np.float32) * scale_ho
        self.b_out = np.zeros((1, self.output_size), dtype=np.float32)
        
    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    def _sigmoid_derivative(self, s):
        """Derivative of sigmoid given sigmoid output."""
        return s * (1 - s)
    
    def _tanh_derivative(self, t):
        """Derivative of tanh given tanh output."""
        return 1 - t ** 2
    
    def _lstm_forward(self, x_seq, return_cache=False):
        """
        Forward pass through LSTM.
        
        Args:
            x_seq: Input sequence of shape (seq_len, input_size)
            return_cache: If True, return intermediate values for backprop
            
        Returns:
            output: Final output
            cache: Intermediate values if return_cache=True
        """
        seq_len = x_seq.shape[0]
        h = self.hidden_size
        
        # Initialize hidden states
        h_t = np.zeros((1, h), dtype=np.float32)
        c_t = np.zeros((1, h), dtype=np.float32)
        
        # Cache for backpropagation
        cache = {
            'x': [], 'h': [h_t.copy()], 'c': [c_t.copy()],
            'i': [], 'f': [], 'g': [], 'o': []
        }
        
        for t in range(seq_len):
            x_t = x_seq[t:t+1, :]  # (1, input_size)
            
            # Compute gates
            gates = x_t @ self.W_ih + h_t @ self.W_hh + self.b_ih + self.b_hh
            
            # Split gates
            i_t = self._sigmoid(gates[:, :h])           # Input gate
            f_t = self._sigmoid(gates[:, h:2*h])        # Forget gate
            g_t = np.tanh(gates[:, 2*h:3*h])            # Cell gate
            o_t = self._sigmoid(gates[:, 3*h:])         # Output gate
            
            # Update cell and hidden state
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * np.tanh(c_t)
            
            # Store cache
            cache['x'].append(x_t)
            cache['h'].append(h_t.copy())
            cache['c'].append(c_t.copy())
            cache['i'].append(i_t)
            cache['f'].append(f_t)
            cache['g'].append(g_t)
            cache['o'].append(o_t)
        
        # Output layer
        output = h_t @ self.W_out + self.b_out
        
        if return_cache:
            return output, cache
        return output
    
    def _lstm_backward(self, d_output, cache):
        """
        Backward pass through LSTM (BPTT - Backpropagation Through Time).
        
        Args:
            d_output: Gradient of loss w.r.t output
            cache: Cached values from forward pass
            
        Returns:
            gradients: Dictionary of gradients for all weights
        """
        h = self.hidden_size
        seq_len = len(cache['x'])
        
        # Initialize gradients
        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db_ih = np.zeros_like(self.b_ih)
        db_hh = np.zeros_like(self.b_hh)
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)
        
        # Output layer gradients
        dW_out = cache['h'][-1].T @ d_output
        db_out = d_output.copy()
        
        # Gradient w.r.t final hidden state
        dh_next = d_output @ self.W_out.T
        dc_next = np.zeros((1, h), dtype=np.float32)
        
        # Backprop through time
        for t in reversed(range(seq_len)):
            x_t = cache['x'][t]
            h_prev = cache['h'][t]
            c_prev = cache['c'][t]
            c_t = cache['c'][t + 1]
            i_t = cache['i'][t]
            f_t = cache['f'][t]
            g_t = cache['g'][t]
            o_t = cache['o'][t]
            
            # Gradient through output gate
            tanh_c = np.tanh(c_t)
            do = dh_next * tanh_c
            do_gate = do * self._sigmoid_derivative(o_t)
            
            # Gradient through cell state
            dc = dh_next * o_t * self._tanh_derivative(tanh_c) + dc_next
            
            # Gradient through forget gate
            df = dc * c_prev
            df_gate = df * self._sigmoid_derivative(f_t)
            
            # Gradient through input gate
            di = dc * g_t
            di_gate = di * self._sigmoid_derivative(i_t)
            
            # Gradient through cell gate
            dg = dc * i_t
            dg_gate = dg * self._tanh_derivative(g_t)
            
            # Concatenate gate gradients
            d_gates = np.concatenate([di_gate, df_gate, dg_gate, do_gate], axis=1)
            
            # Weight gradients
            dW_ih += x_t.T @ d_gates
            dW_hh += h_prev.T @ d_gates
            db_ih += d_gates
            db_hh += d_gates
            
            # Gradient for previous timestep
            dh_next = d_gates @ self.W_hh.T
            dc_next = dc * f_t
        
        return {
            'W_ih': dW_ih, 'W_hh': dW_hh, 
            'b_ih': db_ih, 'b_hh': db_hh,
            'W_out': dW_out, 'b_out': db_out
        }
    
    def _clip_gradients(self, gradients):
        """Clip gradients to prevent exploding gradients."""
        for key in gradients:
            gradients[key] = np.clip(gradients[key], -self.clip_value, self.clip_value)
        return gradients
    
    def _adam_update(self, gradients):
        """Apply Adam optimizer update."""
        self.t += 1
        
        params = {
            'W_ih': self.W_ih, 'W_hh': self.W_hh,
            'b_ih': self.b_ih, 'b_hh': self.b_hh,
            'W_out': self.W_out, 'b_out': self.b_out
        }
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first and second moment estimates
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Bias-corrected estimates
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Write back
        self.W_ih = params['W_ih']
        self.W_hh = params['W_hh']
        self.b_ih = params['b_ih']
        self.b_hh = params['b_hh']
        self.W_out = params['W_out']
        self.b_out = params['b_out']
    
    def _update_running_stats(self, value):
        """Update running statistics for online normalization."""
        self.n_samples += 1
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Welford's online algorithm
        delta = value - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = value - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.n_samples
        
    def _normalize(self, value):
        """Normalize a value using min-max scaling."""
        if self.max_val == self.min_val:
            return 0.5
        return (value - self.min_val) / (self.max_val - self.min_val + 1e-8)
    
    def _denormalize(self, value):
        """Denormalize a value back to original scale."""
        if self.max_val == self.min_val:
            return self.running_mean
        return value * (self.max_val - self.min_val) + self.min_val
    
    def _decay_learning_rate(self):
        """Apply learning rate decay."""
        self.learning_rate = max(
            self.learning_rate * self.decay_rate,
            self.min_learning_rate
        )
    
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
        
        # Prepare training data
        buffer_list = list(self.sequence_buffer)
        X_seq = np.array(buffer_list[:-1], dtype=np.float32).reshape(-1, 1)
        y_target = np.array([[buffer_list[-1]]], dtype=np.float32)
        
        # Forward pass with cache
        output, cache = self._lstm_forward(X_seq, return_cache=True)
        
        # Compute MSE loss
        loss = float(np.mean((output - y_target) ** 2))
        
        # Backward pass
        d_output = 2 * (output - y_target) / output.size  # MSE gradient
        gradients = self._lstm_backward(d_output, cache)
        
        # Clip gradients
        gradients = self._clip_gradients(gradients)
        
        # Apply Adam update (Online Gradient Descent step)
        self._adam_update(gradients)
        
        # Update metrics
        self.total_updates += 1
        self.cumulative_loss += loss
        
        # Decay learning rate
        self._decay_learning_rate()
        
        return loss
    
    def predict(self):
        """
        Predict the next burst time.
        
        Returns:
            Predicted burst time (denormalized) or None if not enough data
        """
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        try:
            # Get last sequence_length values
            buffer_list = list(self.sequence_buffer)
            X_seq = np.array(buffer_list[-self.sequence_length:], dtype=np.float32).reshape(-1, 1)
            
            # Forward pass (no cache needed)
            output = self._lstm_forward(X_seq, return_cache=False)
            
            # Denormalize
            prediction = self._denormalize(float(output[0, 0]))
            
            return max(prediction, 0.0)
            
        except Exception as e:
            print(f"Online LSTM prediction error: {e}")
            return None
    
    def get_average_loss(self):
        """Get average loss across all updates."""
        if self.total_updates == 0:
            return 0.0
        return self.cumulative_loss / self.total_updates
    
    def reset(self):
        """Reset model for a new round."""
        self._init_weights()
        self.sequence_buffer.clear()
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_samples = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.h = np.zeros((1, self.hidden_size))
        self.c = np.zeros((1, self.hidden_size))
        self.total_updates = 0
        self.cumulative_loss = 0.0
        self.learning_rate = self.initial_learning_rate
        self.m = {}
        self.v = {}
        self.t = 0


# Global Online LSTM instance
online_lstm_model = None
online_lstm_lock = threading.Lock()

# Configuration
ONLINE_LSTM_SEQUENCE_LENGTH = 3
ONLINE_LSTM_HIDDEN_SIZE = 8
ONLINE_LSTM_LEARNING_RATE = 0.01
ONLINE_LSTM_MIN_SAMPLES = 3


def initialize_online_lstm():
    """Initialize the Online LSTM model."""
    global online_lstm_model
    with online_lstm_lock:
        if online_lstm_model is None:
            online_lstm_model = OnlineLSTMNumpy(
                sequence_length=ONLINE_LSTM_SEQUENCE_LENGTH,
                hidden_size=ONLINE_LSTM_HIDDEN_SIZE,
                learning_rate=ONLINE_LSTM_LEARNING_RATE
            )
            print(f"Online LSTM (NumPy) initialized: seq_len={ONLINE_LSTM_SEQUENCE_LENGTH}, "
                  f"hidden={ONLINE_LSTM_HIDDEN_SIZE}, lr={ONLINE_LSTM_LEARNING_RATE}")


def online_lstm_update(burst_time):
    """
    Update the Online LSTM with a new burst time observation.
    Called when a process completes.
    """
    global online_lstm_model
    
    initialize_online_lstm()
    
    with online_lstm_lock:
        try:
            loss = online_lstm_model.partial_fit(burst_time)
            if loss is not None:
                print(f"Online LSTM updated: burst_time={burst_time:.4f}, "
                      f"loss={loss:.6f}, updates={online_lstm_model.total_updates}", flush=True)
        except Exception as e:
            print(f"Online LSTM update error: {e}", flush=True)


def get_burst_time_prediction_online_lstm():
    """
    Get burst time prediction using Online LSTM with OGD.
    Falls back to EWMA if not enough samples.
    """
    global online_lstm_model
    
    initialize_online_lstm()
    
    history = processExecutionHistory[FUNCTION_HISTORY_KEY]
    
    if not history:
        return 2.0  # Default
    
    with online_lstm_lock:
        if online_lstm_model.n_samples >= ONLINE_LSTM_MIN_SAMPLES:
            lstm_pred = online_lstm_model.predict()
            
            if lstm_pred is not None and lstm_pred > 0:
                # Blend with EWMA for robustness
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
    """Reset the Online LSTM model."""
    global online_lstm_model
    with online_lstm_lock:
        if online_lstm_model is not None:
            online_lstm_model.reset()
            print("Online LSTM model reset", flush=True)


# ============================================================================
# END OF ONLINE LSTM IMPLEMENTATION
# ============================================================================


# The function to update the core nums by request.
def updateThread():
    global numCores

    myHost = '0.0.0.0'
    myPort = 5500

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind((myHost, myPort))
    serverSocket.listen(1)

    while True:
        (clientSocket, _) = serverSocket.accept()
        data_ = clientSocket.recv(1024)
        dataStr = data_.decode('UTF-8')
        dataStrList = dataStr.splitlines()
        message = json.loads(dataStrList[-1])

        numCores = message["numCores"]
        result = {"Response": "Ok"}
        msg = json.dumps(result)

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

    if numCoreFlag == False:
        result = actionModule.lambda_handler(message)

        turnaround_time = time.time() - arrival_time
        result["turnaround_time"] = turnaround_time

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
    response_status_text = 'OK'

    r = '%s %s %s\r\n' % (response_proto, response_status,
                          response_status_text)
    
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTSTP})
        
        clientSocket_.send(r.encode(encoding="utf-8"))
        clientSocket_.send(response_headers_raw.encode(encoding="utf-8"))
        clientSocket_.send('\r\n'.encode(encoding="utf-8"))
        clientSocket_.send(msg.encode(encoding="utf-8"))
    except Exception as e:
        print(f"Error sending response: {e}", flush=True)
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTSTP})
        try:
            clientSocket_.close()
        except:
            pass


def calculate_ewma(history, alpha=0.8):
    if not history:
        return 0
    ewma = history[0]
    for val in history[1:]:
        ewma = alpha * val + (1 - alpha) * ewma
    return ewma


ALPHA_RT = 0.7
BETA_RT = 0.3


def calculate_remaining_time(pid):
    """Calculate remaining time using Online LSTM prediction."""
    estimated_burst_time = get_burst_time_prediction_online_lstm()
    
    elapsed_time = processExecutedTime.get(pid, 0)
    if pid in processStartTime:
        elapsed_time += time.time() - processStartTime[pid]
    
    remaining_time = max(estimated_burst_time - elapsed_time, 0)
    
    return remaining_time


def calculate_total_wait_time(processQueue):
    total_wait_time = 0
    current_time = time.time()

    for process_item in processQueue:
        _, pid = process_item
        if pid in processTimestamps:
            acc_wait, last_wait = processTimestamps[pid]
            individual_wait = acc_wait + (current_time - last_wait if last_wait else 0.0)
            total_wait_time += individual_wait

    return total_wait_time


def calculate_dynamic_beta(total_wait_time, num_tasks):
    if num_tasks == 0:
        return 0.2

    dynamic_beta = total_wait_time / (num_tasks + 1)

    return min(max(dynamic_beta, 0.1), 1.0)


PREEMPTION_THRESHOLD = 4


def waitTermination(childPid):
    """Wait for process completion and update Online LSTM."""
    global processQueue, mapPIDtoStatus

    os.waitpid(childPid, 0)

    lockPIDMap.acquire()

    try:
        mapPIDtoStatus.pop(childPid, None)

        total_executed = processExecutedTime.get(childPid, 0.0)
        if childPid in processStartTime:
            total_executed += time.time() - processStartTime[childPid]
        
        processExecutionHistory[FUNCTION_HISTORY_KEY].append(total_executed)
        
        # Update Online LSTM with completed process's burst time
        online_lstm_update(total_executed)
        
        processTimestamps.pop(childPid, None)
        processStartTime.pop(childPid, None)
        processExecutedTime.pop(childPid, None)
    except Exception as e:
        print(f"Error removing process {childPid}: {e}")

    if processQueue:
        def is_process_alive(pid):
            try:
                return os.path.exists(f"/proc/{pid}")
            except Exception:
                return False
        
        original_queue_size = len(processQueue)
        processQueue[:] = [
            (remaining_time, pid) for remaining_time, pid in processQueue
            if pid in mapPIDtoStatus and is_process_alive(pid)
        ]
        
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

        def priority_selector(process_item):
            _, pid = process_item
            remaining_time = calculate_remaining_time(pid) + 1e-9

            if pid in processTimestamps:
                acc_wait, last_wait = processTimestamps[pid]
                individual_wait_time = acc_wait + (time.time() - last_wait if last_wait else 0.0)
            else:
                individual_wait_time = 0

            total_wait_time = calculate_total_wait_time(processQueue)
            dynamic_beta = calculate_dynamic_beta(
                total_wait_time, len(processQueue))

            alpha = 0.8

            priority = (alpha * (1 / (remaining_time + 1e-9))) + \
                (dynamic_beta * individual_wait_time)

            return priority

        next_process_candidates = sorted(
            processQueue, key=priority_selector, reverse=True)

        if next_process_candidates:
            _, nextProcess = next_process_candidates[0]
            current_running_pid = None

            for pid, status in mapPIDtoStatus.items():
                if status == "running":
                    current_running_pid = pid
                    break

            if current_running_pid:
                current_remaining = calculate_remaining_time(
                    current_running_pid)
                next_remaining = calculate_remaining_time(nextProcess)

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
                
                # Add Online LSTM stats
                with online_lstm_lock:
                    if online_lstm_model is not None:
                        result["lstm_updates"] = online_lstm_model.total_updates
                        result["lstm_avg_loss"] = online_lstm_model.get_average_loss()
                        result["lstm_samples"] = online_lstm_model.n_samples
                        result["lstm_learning_rate"] = online_lstm_model.learning_rate
                
                msg = json.dumps(result)
                responseFlag = True

            if "Clear" in message:
                responseMapWindows = []
                processExecutionHistory[FUNCTION_HISTORY_KEY] = []
                
                # Reset Online LSTM
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

        waitForRunning = False
        numIsRunning = 0

        lockPIDMap.acquire()
        for child in mapPIDtoStatus.copy():
            if mapPIDtoStatus[child] == "running":
                numIsRunning += 1
        if numIsRunning >= numCores:
            waitForRunning = True

        if len(responseMapWindows) >= 100:
            responseMapWindows.pop(0)

        childProcess = os.fork()
        if childProcess == 0:
            myFunction(data_, clientSocket, time.time())
            os._exit(os.EX_OK)
        else:
            responseMapWindows.append([childProcess, [time.time(), -1]])
            processStartTime[childProcess] = time.time()

            processTimestamps[childProcess] = (0.0, None)
            
            if waitForRunning:
                mapPIDtoStatus[childProcess] = "waiting"
                os.kill(childProcess, signal.SIGTSTP)

                estimated_burst_time = calculate_remaining_time(childProcess)
                heapq.heappush(processQueue, (estimated_burst_time, childProcess))

                processStartTime.pop(childProcess, None)
                acc, _ = processTimestamps.get(childProcess, (0.0, None))
                processTimestamps[childProcess] = (acc, time.time())
            else:
                mapPIDtoStatus[childProcess] = "running"
                requestQueue.append(childProcess)

            lockPIDMap.release()
            
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

    numCores = 8
    os.sched_setaffinity(0, affinity_mask)

    print("Welcome... ", numCores)
    
    # Initialize Online LSTM at startup
    initialize_online_lstm()
    print("Online LSTM with Online Gradient Descent (NumPy) initialized")

    myHost = '0.0.0.0'
    myPort = int(os.environ.get('PORT', 8081))

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind((myHost, myPort))
    serverSocket.listen(10)

    import app
    actionModule = app

    signal.signal(signal.SIGINT, signal_handler)

    phOut = PrintHook()
    phOut.Start(MyHookOut)

    threadUpdate = threading.Thread(target=updateThread)
    threadUpdate.daemon = True
    threadUpdate.start()

    threadIntercept = threading.Thread(target=IOThread)
    threadIntercept.daemon = True
    threadIntercept.start()

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