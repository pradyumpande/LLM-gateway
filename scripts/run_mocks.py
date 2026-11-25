import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MOCKS = [("a", 9000), ("b", 9001), ("c", 9002)]


def main() -> None:
    processes = []
    try:
        for name, port in MOCKS:
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "mock_llm.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    cwd=ROOT,
                    env={**os.environ, "MOCK_NAME": name},
                )
            )
        while True:
            exited = [process for process in processes if process.poll() is not None]
            if exited:
                raise RuntimeError(
                    "A mock server exited unexpectedly; stopping the remaining servers."
                )
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
