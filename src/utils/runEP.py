import subprocess

windows = False

def run_eval_plus(RESULTS_PATH, SUMMARY_PATH, DATASET):

    if windows:
        command = f"wsl docker run -v /mnt/c/Users/CSE2/Desktop/SCoder:/app ganler/evalplus:latest --dataset {DATASET} --samples /app/{RESULTS_PATH} > C:/Users/CSE2/Desktop/SCoder/{SUMMARY_PATH}\n"

        with open("temp.bat", mode="w", encoding="utf-8") as file:
            file.write(command)   
        
        try:
            result = subprocess.run(["temp.bat"], shell=True)
            # Print the output and error (if any)
            print("Output:\n", result.stdout)
            print("Error:\n", result.stderr)
        except Exception as e:
            print("Error Occured")
            print(e)
    else:
        command = f"docker run -v /home/ashraful/prompting/SCoder:/app ganler/evalplus:latest --dataset {DATASET} --samples /app/{RESULTS_PATH} > /home/ashraful/prompting/SCoder/{SUMMARY_PATH}\n"

        with open("temp.sh", mode="w", encoding="utf-8") as file:
            file.write(command)   
        
        try:
            result = subprocess.run([command], shell=True)
            # Print the output and error (if any)
            print("Output:\n", result.stdout)
            print("Error:\n", result.stderr)
        except Exception as e:
            print("Error Occured")
            print(e)
