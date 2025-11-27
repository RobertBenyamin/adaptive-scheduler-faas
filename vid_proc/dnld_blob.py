import socket
import os

import socket
import os
import json

def download_blob_new(blobName):
    myHost = '0.0.0.0'
    myPort = 3333
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.settimeout(5.0)
    try:
        clientSocket.connect((myHost, myPort))
        message = {"blobName": blobName, "operation": "get", "pid": os.getpid()}
        messageStr = json.dumps(message)
        clientSocket.sendall(messageStr.encode(encoding="utf-8"))

        data_ = b''
        while True:
            try:
                chunk = clientSocket.recv(4096)
                if not chunk:
                    break
                data_ += chunk
            except socket.timeout:
                break
    except socket.timeout:
        pass
    finally:
        clientSocket.close()
    return data_

def upload_blob_new(blobName, value):
    myHost = '0.0.0.0'
    myPort = 3333
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.settimeout(5.0)
    try:
        clientSocket.connect((myHost, myPort))
        message = {"blobName": blobName, "operation": "set", "value": value, "pid": os.getpid()}
        messageStr = json.dumps(message)
        clientSocket.sendall(messageStr.encode(encoding="utf-8"))

        data_ = b''
        while True:
            try:
                chunk = clientSocket.recv(4096)
                if not chunk:
                    break
                data_ += chunk
            except socket.timeout:
                break
    except socket.timeout:
        pass
    finally:
        clientSocket.close()
    return data_