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
    os.environ["RECORDINGS_DIR"] = "D:/RoyStudies/Recordings"
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Starting batch verification across all datasets...")
    
    episodes = 150
    skip_training = False
    
    args_to_forward = []
    for arg in sys.argv[1:]:
        if arg == "--skip-training":
            skip_training = True
            args_to_forward.append(arg)
        else:
            try:
                episodes = int(arg)
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
            "--episodes", str(episodes),
            "--max-workers", "4"
        ] + args_to_forward
        
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                
        rc = process.poll()
        if rc != 0:
            print(f"[ERROR] Run failed for dataset {ds} with exit code {rc}")
        else:
            print(f"[SUCCESS] Completed run for dataset {ds}")

if __name__ == "__main__":
    main()
