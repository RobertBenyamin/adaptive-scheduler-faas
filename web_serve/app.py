import os
import time
from storage_helper import download_file, upload_file
from dnld_blob import download_blob_new

TMP_DIR = "/tmp"

def lambda_handler(event):
    blobName = event.get("input_file", "money.txt")
    pid = str(os.getpid())

    base = os.path.basename(blobName)
    name, ext = os.path.splitext(base)
    proc_blob_name = f"{name}_{pid}{ext}"
    local_file_path = os.path.join(TMP_DIR, proc_blob_name)

    try:
        download_blob_new(blobName)
        timeout = 2.0
        t0 = time.time()
        while not os.path.exists(local_file_path) and (time.time() - t0) < timeout:
            time.sleep(0.01)
        if not os.path.exists(local_file_path):
            download_file(blobName, local_file_path)
    except Exception:
        download_file(blobName, local_file_path)
    
    # Read all lines (first line is balance, others are transaction history)
    with open(local_file_path, "r") as moneyF:
        lines = moneyF.readlines()

    # Parse balance from the first line
    try:
        balance = float(lines[0].strip())
    except Exception:
        balance = 0.0

    # Apply transaction history if present
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            action, amount_str = parts
            try:
                amount = float(amount_str)
                if action.lower() == "deposit":
                    balance += amount
                elif action.lower() == "withdraw":
                    balance -= amount
            except Exception:
                continue  # Ignore malformed lines

    # Write new balance and append this transaction
    output_filename = f"transaction-{blobName}"
    local_output_path = os.path.join(TMP_DIR, output_filename)
    with open(output_filename, "w") as new_file:
        new_file.write(f"{balance}\n")
        # Keep previous transaction history
        for line in lines[1:]:
            new_file.write(line)

    upload_file(local_output_path, output_filename)

    try:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
    except:
        pass

    return {"Money": "withdrawn", "balance": balance, "output_file": output_filename}