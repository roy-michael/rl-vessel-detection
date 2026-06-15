import subprocess
import sys
import os

PYTHON = sys.executable
DATASETS = [
    "croatia",          # 2507_1
    "croatia_2307",
    "croatia_2507_2",
    "croatia_2407_1",
    "croatia_2407_2",
    "scooter"
]

def main():
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Starting batch verification across all datasets...")
    
    episodes = 150
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            pass

    for ds in DATASETS:
        print(f"\n=======================================================")
        print(f"  RUNNING PIPELINE FOR DATASET: {ds} ({episodes} episodes)")
        print(f"=======================================================")
        
        cmd = [
            PYTHON, "-m", "scripts.compare_rl_agents",
            "--train-dataset", ds,
            "--eval-datasets", ds,
            "--episodes", str(episodes)
        ]
        
        # We run the command and capture output to prevent terminal flooding
        # while still letting the user see progress.
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # print some lines to terminal to show progress
                # especially training status or reports
                sys.stdout.write(line)
                sys.stdout.flush()
                
        rc = process.poll()
        if rc != 0:
            print(f"[ERROR] Run failed for dataset {ds} with exit code {rc}")
        else:
            print(f"[SUCCESS] Completed run for dataset {ds}")

if __name__ == "__main__":
    main()
