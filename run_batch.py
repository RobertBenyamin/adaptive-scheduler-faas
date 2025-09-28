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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", type=str, help="Base name for output files (e.g. runners5)")
    parser.add_argument("repeat", type=int, help="How many times to repeat the experiment")
    args = parser.parse_args()

    for i in range(1, args.repeat + 1):
        print(f"\n=== RUN {i} of {args.repeat} ===\n")
        # 1. stop.sh
        run_cmd(["bash", "stop.sh"])
        # 2. deploy_only.sh
        run_cmd(["bash", "deploy_only.sh"])
        # 3. deploy_app.sh
        run_cmd(["bash", "deploy_app.sh"])
        # 4. Get kourier IP
        ip = get_kourier_ip()
        print(f"Kourier IP: {ip}")
        # 5. python3 knative.py --target_ip xx.xx.xx.xx
        run_cmd(["python3", "knative.py", "--target_ip", ip])
        # 6. python3 analyze.py --output_file output.xlsx
        output_file = f"{args.basename}-{i}.xlsx"
        run_cmd(["python3", "analyze.py", "--output_file", output_file])
        print(f"=== Finished run {i} ===\n")
        # sleep between runs
        time.sleep(15)

if __name__ == "__main__":
    main()