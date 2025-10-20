import argparse
import subprocess
import sys
import time
import re

def get_kourier_ip():
    cmd = ["kubectl", "get", "svc", "-n", "kourier-system", "kourier", "-o", "wide"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to get kourier service IP")
        print(result.stderr)
        sys.exit(1)
    # Try to extract the EXTERNAL-IP or CLUSTER-IP from the output
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        print("Unexpected kubectl output")
        sys.exit(1)
    header = lines[0].split()
    ip_col = None
    for idx, col in enumerate(header):
        if col in ("EXTERNAL-IP", "CLUSTER-IP"):
            ip_col = idx
            break
    if ip_col is None:
        print("Could not find IP column in kubectl output")
        sys.exit(1)
    ip = lines[1].split()[ip_col]
    # Remove possible <none>
    if ip == "<none>":
        print("Kourier IP is <none>, cannot continue.")
        sys.exit(1)
    return ip

def run_cmd(cmd, shell=False):
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(cmd, shell=shell)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)

def wait_namespace_not_terminating(namespace, timeout=600, poll_interval=5):
    """
    Return True when namespace is absent or its phase is not 'Terminating'.
    Return False if timeout elapsed while still Terminating.
    """
    start = time.time()
    while True:
        result = subprocess.run(["kubectl", "get", "ns", namespace, "-o", "jsonpath={.status.phase}"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            # namespace not found => OK (deleted)
            print(f"Namespace {namespace} not found (treated as deleted).")
            return True
        phase = result.stdout.strip()
        if phase != "Terminating":
            print(f"Namespace {namespace} phase: {phase}")
            return True
        if time.time() - start > timeout:
            print(f"Timeout waiting for namespace {namespace} to exit Terminating.")
            return False
        print(f"Namespace {namespace} is Terminating; waiting {poll_interval}s...")
        time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", type=str, help="Base name for output files (e.g. runners5)")
    parser.add_argument("repeat", type=int, help="How many times to repeat the experiment")
    args = parser.parse_args()

    for i in range(1, args.repeat + 1):
        print(f"\n=== RUN {i} of {args.repeat} ===\n")
        # 1. stop.sh
        run_cmd(["bash", "stop.sh"])
        # Wait until knative-serving namespace is not Terminating (or gone)
        ok = wait_namespace_not_terminating("knative-serving", timeout=600)
        if not ok:
            print("knative-serving namespace still terminating after timeout. You can:")
            print("  - increase the timeout,")
            print("  - remove finalizers manually (kubectl replace --raw /api/v1/namespaces/knative-serving/finalize -f <edited-json>),")
            print("  - or inspect 'kubectl describe ns knative-serving' to find blockers.")
            sys.exit(1)
        time.sleep(10)
        # 2. deploy_only.sh
        run_cmd(["bash", "deploy_only.sh"])
        time.sleep(30)
        # 3. deploy_app.sh
        run_cmd(["bash", "deploy_app.sh"])
        time.sleep(30)
        # 4. Get kourier IP
        ip = get_kourier_ip()
        print(f"Kourier IP: {ip}")
        time.sleep(30)
        # 5. python3 knative.py --target_ip xx.xx.xx.xx
        run_cmd(["python3", "knative.py", "--target_ip", ip])
        time.sleep(30)
        # 6. python3 analyze.py --output_file output.xlsx
        output_file = f"{args.basename}-{i}.xlsx"
        run_cmd(["python3", "analyze.py", "--output_file", output_file])
        print(f"=== Finished run {i} ===\n")
        # sleep between runs
        time.sleep(30)

if __name__ == "__main__":
    main()