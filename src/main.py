# main.py
import subprocess

scripts = [
    "src/preprocessing.py",
    "src/language_model.py",
    "src/naive_bayes.py",
    "src/ir_system.py"
]

print("--- Starting NLP News Analysis Pipeline ---")
for script in scripts:
    print(f"\nExecuting: {script}...")
    subprocess.run(["python", script])
    print(f"Finished: {script}")
print("\n--- Pipeline Complete ---")
