import argparse
import random
import subprocess
import time
import numpy as np
import threading
import requests
from statistics import mean, median

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
    "ml-train":    [f"dataset{i}.csv" for i in range(1, 11)],
    "web-serve":   [f"account{i}.txt" for i in range(1, 11)],
}

# --- IMPROVED: More varied and unpredictable test sequences ---
# Pattern designed to challenge SRTF scheduling:
# - Create backlogs with large jobs
# - Test prioritization with bursts of small jobs
# - Mix medium jobs to add unpredictability
# - Total 18 items to avoid too much repetition
TEST_SEQUENCES = {
    "cnn-serving": [f"img{i}.jpg" for i in [
        41, 40,
        2, 1, 3,
        20, 25, 28,
    ]],
    "img-rot": [f"img{i}.jpg" for i in [
        41, 40,
        2, 1, 3,
        20, 25, 28,
    ]],
    "img-res": [f"img{i}.jpg" for i in [
        41, 40,
        2, 1, 3,
        20, 25, 28,
    ]],
    "vid-proc": [f"vid{i}.mp4" for i in [
        10, 9,
        1, 2, 3,
        5, 6, 7
    ]],
    "ml-train": [f"dataset{i}.csv" for i in [
        10, 9,
        1, 2, 3,
        5, 6, 7
    ]],
    "web-serve": [f"account{i}.txt" for i in [
        10, 9,
        1, 2, 3,
        5, 6, 7
    ]],
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

def lambda_func(service, service_name, request_index, runner_times_list):
    global times
    t1 = time.time()

    # Add custom Host header (virtual host routing)
    headers = {
        "Host": service.replace("http://", ""),  # example: "cnn-serving.default.52.72.211.10.nip.io"
        "Content-Type": "application/json"
    }

    payload = {}

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

def warmup_phase(service, service_name, num_warmup=25):
    """
    Send initial requests to build history for prediction models.
    This ensures both the mean-based and your proposed method have sufficient data.
    """
    print(f"Starting warmup phase for {service_name} with {num_warmup} requests...")
    warmup_times = []
    
    for i in range(num_warmup):
        lambda_func(service, service_name, i, warmup_times)
        time.sleep(0.2)  # small delay between warmup requests
    
    print(f"Warmup complete for {service_name}. History built with {len(warmup_times)} requests.")

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

loads = [5, 25, 30]
load_desc = ["LOW_LOAD", "MED_LOAD", "HIGH_LOAD"]

output_file = open("run-all-out.txt", "w")

indR = 0
for load in loads:
    duration = 2  # Changed from 2 to 5 seconds
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
        
        # Warmup phase: build initial history for prediction models
        # warmup_phase(service, current_service_name, num_warmup=5)

        st = 0
  
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