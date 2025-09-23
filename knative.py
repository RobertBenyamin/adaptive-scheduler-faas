import subprocess
import time
import numpy as np
import threading
import requests
from statistics import mean, median,variance,stdev

# get the url of a function
def getUrlByFuncName(funcName):
    try:
        output = subprocess.check_output("kn service describe " + funcName + " -vvv", shell=True).decode("utf-8")
    except Exception as e:
        print("Error in kn service describe == " + str(e))
        return None
    lines = output.splitlines()
    for line in lines:
        if "URL:"  in line:
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

def lambda_func(service):
    global times
    t1 = time.time()

    # IP address of the ingress (could be LoadBalancer, NodePort, etc.)
    # use this command to get IP -> sudo kubectl get svc -n kourier-system kourier -o wide
    target_ip = "http://10.43.251.100"  # or whatever the IP is

    # Add custom Host header (virtual host routing)
    headers = {
        "Host": service.replace("http://", ""),  # example: "cnn-serving.default.52.72.211.10.nip.io"
        "Content-Type": "application/json"
    }

    # Perform POST to the IP, but override the Host header
    r = requests.post(target_ip, headers=headers, json={"name": "test"})

    print(r.text)
    t2 = time.time()
    times.append(t2 - t1)

def EnforceActivityWindow(start_time, end_time, instance_events):
    events_iit = []
    events_abs = [0] + instance_events
    event_times = [sum(events_abs[:i]) for i in range(1, len(events_abs) + 1)]
    event_times = [e for e in event_times if (e > start_time)and(e < end_time)]
    try:
        events_iit = [event_times[0]] + [event_times[i]-event_times[i-1]
                                         for i in range(1, len(event_times))]
    except:
        pass
    return events_iit

loads = [2, 5, 15]
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
    instance_events = EnforceActivityWindow(0,duration,inter_arrivals)
        
    for service in services:
        
        threads = []
        times = []
        after_time, before_time = 0, 0

        st = 0
        for t in instance_events:
            st = st + t - (after_time - before_time)
            before_time = time.time()
            if st > 0:
                time.sleep(st)

            threadToAdd = threading.Thread(target=lambda_func, args=(service, ))
            threads.append(threadToAdd)
            threadToAdd.start()
            after_time = time.time()

        for thread in threads:
            thread.join()

        print("=====================" + serviceNames[services.index(service)] + f" with {load_desc[loads.index(load)]}" + "=====================", file=output_file, flush=True)
        print(mean(times), file=output_file, flush=True)
        print(median(times), file=output_file, flush=True)
        print(np.percentile(times, 90), file=output_file, flush=True)
        print(np.percentile(times, 95), file=output_file, flush=True)
        print(np.percentile(times, 99), file=output_file, flush=True)

        print(f"all times note for {load_desc[loads.index(load)]}", file=output_file, flush=True)
        print(times, file=output_file, flush=True)

        