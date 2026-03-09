import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# === CONFIG ===
LOG_DIR = "./tensorboard_logs_sac"   # Path to your logs
OUT_DIR = "./exported_csv"           # Folder to save CSVs

os.makedirs(OUT_DIR, exist_ok=True)

# Find all event files
event_files = []
for root, dirs, files in os.walk(LOG_DIR):
    for f in files:
        if f.startswith("events.out"):
            event_files.append(os.path.join(root, f))

print(f"Found {len(event_files)} event files")

# Process each event file
for file_path in event_files:
    print(f"Processing: {file_path}")
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()

    # Extract scalar tags
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        df = pd.DataFrame([(e.wall_time, e.step, e.value) for e in events],
                          columns=["wall_time", "step", "value"])
        out_path = os.path.join(OUT_DIR, f"{tag.replace('/', '_')}.csv")
        df.to_csv(out_path, index=False)
        print(f" Saved: {out_path}")

print(" All scalar logs exported successfully!")
