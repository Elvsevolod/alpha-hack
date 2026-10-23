"""Start a private-network relay without publishing any host port."""
from pathlib import Path
import os
import subprocess

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
    if not os.environ.get(name):
        raise ValueError("Required model channel setting is absent")
command = ["docker", "run", "-d", "--name", "alpha-v010-private-relay", "--network", "alpha-v010-llm",
    "--ip", "192.168.97.2", "--cpus", ".5", "--memory", "256m",
    "-v", str(ROOT / "tools/qwen_channel_proxy.py") + ":/relay.py:ro",
    "--env", "OPENAI_API_KEY", "--env", "OPENAI_BASE_URL", "qwen-ml-agent:exact", "python", "/relay.py",
    "--port", "18765", "--bind", "192.168.97.2"]
subprocess.run(command, env=os.environ.copy(), check=True)
subprocess.run(["docker", "network", "connect", "bridge", "alpha-v010-private-relay"], check=True)
print("Relay listens only on the isolated internal network; no host ports published")
