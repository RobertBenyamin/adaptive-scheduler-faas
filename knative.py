import argparse
import random
import subprocess
import time
import numpy as np
import threading
import requests
from statistics import mean, median, variance, stdev


# IP address of the ingress (could be LoadBalancer, NodePort, etc.)
# use this command to get IP -> sudo kubectl get svc -n kourier-system kourier -o wide
# python3 knative.py --target_ip 10.43.251.105
parser = argparse.ArgumentParser()
parser.add_argument('--target_ip', type=str, default="10.43.251.105", help='Ingress IP address only (e.g., 10.43.251.105)')
args = parser.parse_args()
target_ip = f"http://{args.target_ip}"

# get the url of a function
def getUrlByFuncName(funcName):
    try:
        output = subprocess.check_output("kn service describe " + funcName + " -vvv", shell=True).decode("utf-8")
    except Exception as e:
        print("Error in kn service describe == " + str(e))
        return None
    lines = output.splitlines()
    for line in lines:
        if "URL:" in line:
            url = line.split()[1]
            return url

output = subprocess.check_output("kn service list", shell=True).decode("utf-8")
lines = output.splitlines()
lines = lines[1:] # delete the first line

services = []
serviceNames = []

for line in lines:
    serviceName = line.split()[0]
    if serviceName not in serviceNames:
        serviceNames.append(serviceName)

for serviceName in serviceNames:
    services.append(getUrlByFuncName(serviceName))

# These files must be uploaded to your S3 bucket.
TEST_DATA_CONFIG = {
    "cnn-serving": [f"img{i}.jpg" for i in range(1, 41)],
    "img-rot":     [f"img{i}.jpg" for i in range(1, 41)],
    "img-res":     [f"img{i}.jpg" for i in range(1, 41)],
    "vid-proc":    [f"vid{i}.mp4" for i in range(1, 11)],
    "ml-train":    [f"dataset{i}.csv" for i in range(1, 6)],
    "web-serve":   [f"account{i}.txt" for i in range(1, 11)],
}

# --- NEW: Deliberate Test Sequences ---
# A dictionary of deliberate, non-random request patterns.
# This creates a challenging scenario to test the scheduler's ability
# to prioritize short jobs when long jobs are also present.
# The pattern is generally [long, long, long, short, short, long] to create a "traffic jam".
TEST_SEQUENCES = {
    "cnn-serving": [f"img{i}.jpg" for i in [40, 38, 3, 2, 1, 30, 20]],
    "img-rot":     [f"img{i}.jpg" for i in [40, 38, 3, 2, 1, 30, 20]],
    "img-res":     [f"img{i}.jpg" for i in [40, 38, 3, 2, 1, 30, 20]],
    "vid-proc":     [f"vid{i}.mp4" for i in [10, 9, 2, 1, 7, 6]],
    "ml-train":    [f"dataset{i}.csv" for i in [5, 1, 2, 3, 4]],
    "web-serve":   [f"account{i}.txt" for i in [10, 9, 2, 1, 7, 6]],
}

def reset_server_state(service):
    """
    Send a Clear signal to the server to reset its state before each test round.
    This ensures that execution history and burst time predictions don't carry over between rounds.
    """
    headers = {
        "Host": service.replace("http://", ""),
        "Content-Type": "application/json"
    }
    payload = {"Clear": True}
    try:
        r = requests.post(target_ip, headers=headers, json=payload, timeout=5)
        print(f"Reset server state for {service}: status={r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not reset server state for {service}: {e}")

# --- MODIFIED FUNCTION SIGNATURE: Added 'request_index' ---
def lambda_func(service, service_name, request_index, runner_times_list):
    global times
    t1 = time.time()

    # Add custom Host header (virtual host routing)
    headers = {
        "Host": service.replace("http://", ""),  # example: "cnn-serving.default.52.72.211.10.nip.io"
        "Content-Type": "application/json"
    }

    # --- MODIFIED PAYLOAD CREATION ---
    payload = {}
    # Use the deliberate test sequence if available for the service
    if service_name in TEST_SEQUENCES:
        sequence = TEST_SEQUENCES[service_name]
        # Cycle through the sequence if the number of requests exceeds its length
        input_file = sequence[request_index % len(sequence)]
        payload = {"input_file": input_file}
    # Fallback to random choice if not in the deliberate test sequences
    elif service_name in TEST_DATA_CONFIG:
        possible_inputs = TEST_DATA_CONFIG[service_name]
        input_file = random.choice(possible_inputs)
        payload = {"input_file": input_file}
    else:
        # Default payload if service is not in our config
        payload = {"name": "test"}

    # Perform POST to the IP, but override the Host header
    try:
        r = requests.post(target_ip, headers=headers, json=payload)
        # safer debug output
        print(f"status={r.status_code} url={service} body={repr(r.text)}")

        try:
            response_json = r.json()
            if "turnaround_time" in response_json:
                runner_times_list.append(response_json["turnaround_time"])
        except Exception:
            # If parsing fails or key is missing, just continue
            pass
    except requests.exceptions.RequestException as e:
        # network / connection error
        print(f"request error for {service}: {e}")
        r = None

    t2 = time.time()
    times.append(t2 - t1)

def EnforceActivityWindow(start_time, end_time, instance_events):
    events_iit = []
    events_abs = [0] + instance_events
    event_times = [sum(events_abs[:i]) for i in range(1, len(events_abs) + 1)]
    event_times = [e for e in event_times if (e > start_time) and (e < end_time)]
    try:
        events_iit = [event_times[0]] + [event_times[i]-event_times[i-1]
                                         for i in range(1, len(event_times))]
    except:
        pass
    return events_iit

loads = [2, 5, 15] # Changed from [5, 20, 50]
load_desc = ["LOW_LOAD", "MED_LOAD", "HIGH_LOAD"]

output_file = open("run-all-out.txt", "w")

indR = 0
for load in loads:
    duration = 2
    seed = 100
    rate = load
    # generate Poisson's distribution of events
    inter_arrivals = []
    np.random.seed(seed)
    beta = 1.0/rate
    oversampling_factor = 2
    inter_arrivals = list(np.random.exponential(scale=beta, size=int(oversampling_factor*duration*rate)))
    instance_events = EnforceActivityWindow(0, duration, inter_arrivals)

    for service in services:
        
        threads = []
        times = []
        runner_times = []
        after_time, before_time = 0, 0

        # Get the service name from the service URL
        current_service_name = serviceNames[services.index(service)]

        # Reset server state before each test round to separate metrics between rounds
        reset_server_state(service)

        st = 0
        # --- MODIFIED REQUEST GENERATION LOOP ---
        # Use enumerate to get the index of each request
        for i, t in enumerate(instance_events):
            st = st + t - (after_time - before_time)
            before_time = time.time()
            if st > 0:
                time.sleep(st)

            # Pass the request index 'i' to the lambda_func
            threadToAdd = threading.Thread(target=lambda_func, args=(service, current_service_name, i, runner_times))
            threads.append(threadToAdd)
            threadToAdd.start()
            after_time = time.time()

        for thread in threads:
            thread.join()

        print("=====================" + serviceNames[services.index(service)] + f" with {load_desc[loads.index(load)]}" + "=====================", file=output_file, flush=True)
        print(mean(runner_times), file=output_file, flush=True)
        print(median(runner_times), file=output_file, flush=True)
        print(np.percentile(runner_times, 90), file=output_file, flush=True)
        print(np.percentile(runner_times, 95), file=output_file, flush=True)
        print(np.percentile(runner_times, 99), file=output_file, flush=True)

        print(f"all times note for {load_desc[loads.index(load)]}", file=output_file, flush=True)
        print(runner_times, file=output_file, flush=True)

