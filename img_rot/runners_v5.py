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
processExecutionHistory = {}  # Menyimpan histori eksekusi proses
processStartTime = {}


lockPIDMap = threading.Lock()
requestQueue = []  # queue of child processes
mapPIDtoStatus = {}  # map from pid to status (running, waiting)

processArrivalTimes = {}  # Dictionary to track arrival times of processes
responseMapWindows = []  # map from pid to response

affinity_mask = {0, 1, 2, 3, 4, 5, 6, 7}


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
    if not history:
        return 0  # Jika tidak ada data, kembalikan 0
    ewma = history[0]  # Nilai awal
    for val in history[1:]:
        ewma = alpha * val + (1 - alpha) * ewma
    return ewma

# Placeholder Model Training (Random Forest & Linear Regression)


def train_models(history):
    if len(history) < 5:  # Butuh minimal 5 data untuk regresi
        return np.mean(history), np.mean(history)

    X = np.array(range(len(history))).reshape(-1, 1)
    y = np.array(history)

    # Model Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    lin_pred = lin_model.predict([[len(history)]])[
        0]  # Prediksi waktu berikutnya

    # Model Random Forest
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict([[len(history)]])[
        0]  # Prediksi waktu berikutnya

    return lin_pred, rf_pred


# Parameter Mitigasi Ketidakpastian
ALPHA_RT = 0.7  # Faktor koreksi waktu estimasi
BETA_RT = 0.3   # Faktor penalti standar deviasi


# Fungsi Menghitung Remaining Time
def calculate_remaining_time(pid):
    """
    Menghitung sisa waktu eksekusi berdasarkan average burst time dari histori eksekusi.
    """
    if pid not in processExecutionHistory:
        # Gunakan initial burst time jika belum ada histori
        # Default 2 detik jika tidak ditemukan
        initial_burst, _ = processTimestamps.get(pid, (2, time.time()))
        return initial_burst

    history = processExecutionHistory[pid]

    # Menghitung rata-rata burst time dari histori
    avg_burst_time = sum(history) / \
        len(history) if history else processTimestamps[pid][0]

    # Hitung waktu yang sudah berjalan
    elapsed_time = time.time() - processStartTime.get(pid, time.time())

    # Estimasi sisa waktu
    remaining_time = max(avg_burst_time - elapsed_time, 0)

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

# Dictionary untuk menyimpan waktu mulai eksekusi setiap proses


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

            if childPid not in processExecutionHistory:
                processExecutionHistory[childPid] = []

            processExecutionHistory[childPid].append(elapsed)

    except Exception as e:
        print(f"Error removing process {childPid}: {e}")

    # PREEMPTION: Cek apakah ada proses dengan waktu tersisa lebih pendek dari proses yang berjalan
    if processQueue:
        # Urutkan queue berdasarkan 1 / remaining_time untuk SRTF
        def priority_selector(process_item):
            _, pid = process_item
            remaining_time = calculate_remaining_time(pid) + 1e-9

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
                current_remaining = calculate_remaining_time(
                    current_running_pid)
                next_remaining = calculate_remaining_time(nextProcess)

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

    # blob_client = BlobClient.from_connection_string(connection_string, container_name="artifacteval", blob_name=blobName)

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
            # blob_val = (blob_client.download_blob()).readall()
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
        # blob_client.upload_blob(value, overwrite=True)
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


# Parameters for aging and starvation
agingFactor = 0.1  # Decrease burst time by 0.1 second for every second of waiting
MAX_WAIT_TIME = 30  # seconds, after which process will be promoted to running

# Function to adjust priorities based on aging


def adjustPriorityAging():
    currentTime = time.time()
    updatedQueue = []
    while processQueue:
        burstTime, pid = heapq.heappop(processQueue)
        waitTime = currentTime - processArrivalTimes[pid]
        # Decrease burstTime based on waitTime (aging factor)
        adjustedBurstTime = max(burstTime - (waitTime * agingFactor), 0)
        heapq.heappush(updatedQueue, (adjustedBurstTime, pid))

    # Replace old queue with updated one
    processQueue[:] = updatedQueue

# Function to handle starvation by promoting long-waiting processes


def handleStarvation():
    currentTime = time.time()
    lockPIDMap.acquire()
    try:
        for burstTime, pid in processQueue:
            waitTime = currentTime - processArrivalTimes[pid]
            if waitTime >= MAX_WAIT_TIME:
                # Force this process to run by promoting its priority
                mapPIDtoStatus[pid] = "running"
                os.kill(pid, signal.SIGCONT)  # Resume process
                requestQueue.append(pid)
                break  # Handle one process at a time
    except:
        pass
    finally:
        lockPIDMap.release()


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
    myPort = int(os.environ.get('PORT', 9999))

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
