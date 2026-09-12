"""Record process activity beside power windows, without collecting arguments."""

import json
import signal
import subprocess
import sys
import threading
import time


def capture(output):
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    with open(output, "w") as stream:
        while not stopped.is_set():
            start = time.time_ns() // 1_000_000
            result = subprocess.run(
                ["/bin/ps", "-axo", "pid=,ppid=,%cpu=,time=,comm="],
                capture_output=True, text=True, timeout=5, check=True,
            )
            # Keep every process: a fixed suspect list missed mediaanalysisd.
            # CPU time permits interval comparisons; ps %cpu is a decayed
            # average, not CPU utilization confined to this sample window.
            processes = []
            for line in result.stdout.splitlines():
                pid, ppid, cpu, cpu_time, command = line.split(None, 4)
                processes.append(dict(pid=int(pid), ppid=int(ppid),
                                      cpu_percent=float(cpu), cpu_time=cpu_time,
                                      command=command))
            print(json.dumps(dict(start_unix_ms=start,
                                  end_unix_ms=time.time_ns() // 1_000_000,
                                  processes=processes)), file=stream, flush=True)
            stopped.wait(1)


if __name__ == "__main__":
    capture(sys.argv[1])
