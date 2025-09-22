import subprocess
import time
import requests
import numpy as np
import threading
from statistics import mean, median,variance,stdev

def get_pods():
    # Run the kubectl command
    result = subprocess.run(
        ["kubectl", "get", "pods", "-o", "wide"],
        capture_output=True,
        text=True
    )
    return result.stdout

def count_pods(pods_output):
    # Split the output by lines
    lines = pods_output.strip().split('\n')
    # The first line is the header, so subtract 1 from the total count
    return len(lines) - 1


def parse_pods(pods_output):
    # Split the output by lines
    lines = pods_output.strip().split('\n')
    # Extract the pod names from the lines, skipping the header line
    pod_names = [line.split()[0] for line in lines[1:]]

    start_port = 8001
    pod_dict = {}
    for i, pod_name in enumerate(pod_names):
        pod_dict[pod_name] = start_port + i

    return pod_dict

# Dictionary to store pod names and their port-forward processes
port_forward_processes = {}

def port_forward(pod_name, port):
    # Run the kubectl port-forward command
    command = ["kubectl", "port-forward", f"pod/{pod_name}", f"{port}:8001"]
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command)
    # Wait a bit to ensure the port-forward is established
    time.sleep(2)
    # Store the process in the dictionary
    port_forward_processes[pod_name] = process

# Lambda function to be tested
def lambda_func(service, times):
    # global times
    t1 = time.time()
    r = requests.post(service, json={"name": "test"})
    print(r.text)
    t2 = time.time()
    times.append(t2 - t1)
    # return t2-t1

# do the same with previous code
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

def stop_port_forward(pod_name):
    # Get the process associated with the pod_name
    process = port_forward_processes.get(pod_name)
    if process:
        # Terminate the process
        process.terminate()
        print(f"Terminated port-forward process for pod {pod_name}")
        # Remove the process from the dictionary
        del port_forward_processes[pod_name]
    else:
        print(f"No port-forward process found for pod {pod_name}")

def main():

    # Get the pod details
    pods_output = get_pods()
    # print("Pods output:\n", pods_output)
    
    # Count the number of pods
    pod_count = count_pods(pods_output)
    print(f"Number of pods: {pod_count}")

    # Parse the pod names
    pod_dict = parse_pods(pods_output)
    print("List of pod and its port :", pod_dict)

    services = []

    # Port-forward each pod and keep track of processes
    for pod_name, port in pod_dict.items():
        times = []
        process = port_forward(pod_name, port)

        service_url = f"http://localhost:{port}"
        print(f"Testing pod with IP: {service_url} with pod name is {pod_name}")
        services.append(service_url)
        
        # Print the recorded times
        # print(f"getting responses with {waktu} second")
    
    # define load for bursting app
    loads = [1, 5]
    load_desc = ["LOW_LOAD", "MED_LOAD"]

    output_file = open("run-all-out.txt", "w")

    # do the same with previous code
    indR = 0
    for load in loads:
        duration = 1
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

                threadToAdd = threading.Thread(target=lambda_func, args=(service, times))
                threads.append(threadToAdd)
                threadToAdd.start()
                after_time = time.time()

            for thread in threads:
                thread.join()

            service_name = [i for i in pod_dict]

            print("=====================" + service_name[services.index(service)] + load_desc[loads.index(load)] +"=====================", file=output_file, flush=True)
            print("first output")
            print(mean(times), file=output_file, flush=True)
            print(median(times), file=output_file, flush=True)
            print(np.percentile(times, 90), file=output_file, flush=True)
            print(np.percentile(times, 95), file=output_file, flush=True)
            print(np.percentile(times, 99), file=output_file, flush=True)
            print("last output")
        
        time.sleep(2)

    # Port-forward each pod and keep track of processes
    for pod_name, _ in pod_dict.items():
        stop_port_forward(pod_name)
        print(f"finish stop port forward {pod_name}")

if __name__ == "__main__":
    main()
