import os
from storage_helper import download_file, upload_file


current_path = "/app/pythonAction"

def lambda_handler(event):
    blobName = event.get("input_file", "money.txt")
    download_file(blobName, f"{current_path}/{blobName}")
    
    # Read all lines (first line is balance, others are transaction history)
    with open(f"{current_path}/{blobName}", "r") as moneyF:
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
    temp_file_path = f"{current_path}/moneyTemp{os.getpid()}.txt"
    with open(temp_file_path, "w") as new_file:
        new_file.write(f"{balance}\n")
        # Keep previous transaction history
        for line in lines[1:]:
            new_file.write(line)

    upload_file(temp_file_path, blobName)

    return {"Money": "withdrawn"}