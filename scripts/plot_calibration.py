import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def main():
    log_file = "outputs/bank_history.jsonl"
    if not os.path.exists(log_file):
        print(f"Log file not found at {log_file}")
        return

    # history[key] = ([attempts], [widths])
    history = defaultdict(lambda: ([], []))

    with open(log_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Make key string representation
            key = str(data["key"])
            attempts = data["total_attempts"]
            lo, hi = data["credible_interval"]
            width = hi - lo
            
            history[key][0].append(attempts)
            history[key][1].append(width)

    if not history:
        print("No data found in log file.")
        return

    plt.figure(figsize=(10, 6))
    for key, (attempts, widths) in history.items():
        plt.plot(attempts, widths, label=key, marker='o', markersize=3, alpha=0.7)

    plt.title("Credible Interval Width vs. Trial Count per Belief Key")
    plt.xlabel("Total Attempts")
    plt.ylabel("Credible Interval Width")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_file = "outputs/calibration_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
