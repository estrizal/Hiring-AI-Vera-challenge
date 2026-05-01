import subprocess, os, time, sys

print('Restarting server...')
subprocess.run("FOR /F \"tokens=5\" %P IN ('netstat -a -n -o ^| findstr :8080') DO TaskKill.exe /PID %P /F", shell=True)
time.sleep(2)

server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--port", "8080", "--env-file", ".env"],
    stdout=subprocess.DEVNULL, 
    stderr=subprocess.DEVNULL
)
time.sleep(3)

print('Running judge...')
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

with open('judge_output_raw.txt', 'w', encoding='utf-8') as f:
    subprocess.run(
        [sys.executable, r"The AI challenge information\judge_simulator.py"], 
        env=env, 
        stdout=f, 
        stderr=subprocess.STDOUT
    )

print('Done!')
