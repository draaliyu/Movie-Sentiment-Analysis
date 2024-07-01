import subprocess
import sys

def run_script(script_name):
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"Output from {script_name}:\n{result.stdout}")

if __name__ == "__main__":
    # List of scripts to run in order
    scripts = [
        'load_data.py',
        'preprocess_data.py',
        'prepare_data.py',
        'train_evaluate.py',
        'plots.py'
    ]

    for script in scripts:
        run_script(script)
