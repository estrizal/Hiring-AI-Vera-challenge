"""
auto_tune.py — autonomous judge runner + score parser.
Run this, read the scores, modify layer3_composer.py, repeat.
"""
import subprocess, re, sys, json, os
from pathlib import Path
from datetime import datetime

JUDGE_DIR = Path(r"c:\ADITYA THINGS\my python projects\Machine Learning\Hiring AI Vera challenge\The AI challenge information")
LOG_FILE  = Path(r"c:\ADITYA THINGS\my python projects\Machine Learning\Hiring AI Vera challenge\tune_log.jsonl")

def run_judge():
    # Restart the server to clear in-memory state (daily message caps)
    print("  Restarting server on port 8080...")
    subprocess.run("FOR /F \"tokens=5\" %P IN ('netstat -a -n -o ^| findstr :8080') DO TaskKill.exe /PID %P /F", shell=True, capture_output=True)
    import time; time.sleep(2)
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8080", "--env-file", ".env"], cwd=JUDGE_DIR.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3) # Wait for startup
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "judge_simulator.py"],
        cwd=JUDGE_DIR, capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace", env=env
    )
    return result.stdout + result.stderr

def parse(output):
    messages = []
    for block in re.findall(r"Message: \"(.+?)\".*?TOTAL: (\d+)/50", output, re.DOTALL):
        msg_preview, total = block
        dims = re.findall(r"(Specificity|Category Fit|Merchant Fit|Decision Quality|Engagement)\s+\[.*?\]\s+(\d+)/10",
                          output[output.find(msg_preview[:20]):output.find(msg_preview[:20])+800])
        messages.append({
            "preview": msg_preview[:60],
            "total": int(total),
            "dims": {k: int(v) for k, v in dims},
        })

    avg_block = re.search(
        r"Avg Specificity.*?(\d+)/10.*?Avg Category Fit.*?(\d+)/10.*?"
        r"Avg Merchant Fit.*?(\d+)/10.*?Avg Decision Quality.*?(\d+)/10.*?"
        r"Avg Engagement.*?(\d+)/10.*?AVERAGE SCORE: (\d+)/50 \((\d+)%\)",
        output, re.DOTALL)

    summary = {}
    if avg_block:
        s,cf,mf,dq,e,total,pct = avg_block.groups()
        summary = {"spec":int(s),"cf":int(cf),"mf":int(mf),"dq":int(dq),"eng":int(e),
                   "total":int(total),"pct":int(pct),"n":len(messages)}

    return messages, summary

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running judge...")
    out = run_judge()
    msgs, summary = parse(out)

    print(f"\n{'='*60}")
    print(f"SCORED: {summary.get('n',0)} messages | AVG: {summary.get('total','?')}/50 ({summary.get('pct','?')}%)")
    print(f"  Spec:{summary.get('spec','?')}  CF:{summary.get('cf','?')}  MF:{summary.get('mf','?')}  DQ:{summary.get('dq','?')}  Eng:{summary.get('eng','?')}")
    print(f"{'='*60}")
    for m in msgs:
        d = m['dims']
        print(f"  [{m['total']:2d}/50] {m['preview'][:55]}")
        print(f"        S={d.get('Specificity','?')} CF={d.get('Category Fit','?')} MF={d.get('Merchant Fit','?')} DQ={d.get('Decision Quality','?')} E={d.get('Engagement','?')}")

    # Identify lowest-scoring message and its weakest dimension
    if msgs:
        worst = min(msgs, key=lambda x: x['total'])
        all_dims = {}
        for m in msgs:
            for k,v in m['dims'].items():
                all_dims.setdefault(k,[]).append(v)
        avg_dims = {k: sum(v)/len(v) for k,v in all_dims.items()}
        weakest = min(avg_dims, key=avg_dims.get)
        print(f"\n  Weakest dim: {weakest} (avg {avg_dims[weakest]:.1f})")
        print(f"  Worst msg [{worst['total']}/50]: {worst['preview'][:55]}")

    # Append to log
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), "summary": summary, "messages": msgs}) + "\n")

    print(f"\n  Log: {LOG_FILE}")
