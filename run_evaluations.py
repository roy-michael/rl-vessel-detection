import os
import subprocess
import asyncio

datasets = ["croatia_2507_2", "croatia_2407_1", "croatia_2407_2", "croatia_2307", "scooter"]
rl_agent = "double_q_learning"
policy_dataset = "croatia"

async def run_evaluations():
    print(f"Starting evaluations for policy trained on {policy_dataset} using {rl_agent}...")
    for ds in datasets:
        print(f"\n--- Evaluating on dataset: {ds} ---")
        cmd = [
            ".venv\\Scripts\\python", "-u", "vessel_tracker_rl.py",
            "--dataset", ds,
            "--rl-agent", rl_agent,
            "--policy-dataset", policy_dataset,
            "--headless"
        ]
        
        # Use subprocess to run it
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            print(line.decode('utf-8').rstrip())
            
        await process.wait()
        if process.returncode == 0:
            print(f"Successfully evaluated on {ds}.")
        else:
            print(f"Error evaluating on {ds}.")

if __name__ == "__main__":
    asyncio.run(run_evaluations())
