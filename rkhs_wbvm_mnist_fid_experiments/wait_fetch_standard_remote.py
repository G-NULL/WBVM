import os
import posixpath
import stat
import time
from pathlib import Path
from typing import Tuple

import paramiko


HOST = "connect.westb.seetacloud.com"
PORT = 47830
USER = "root"
REMOTE_DIR = "/root/rkhs_wbvm_mnist_fid_experiments"
LOCAL_DIR = Path(__file__).resolve().parent
POLL_SECONDS = int(os.environ.get("REMOTE_POLL_SECONDS", "120"))
MAX_WAIT_SECONDS = int(os.environ.get("REMOTE_MAX_WAIT_SECONDS", str(8 * 60 * 60)))


def connect() -> paramiko.SSHClient:
    password = os.environ["SEETA_PASS"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def capture(client: paramiko.SSHClient, command: str) -> Tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=30)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def status_command() -> str:
    return f"""
cd {REMOTE_DIR}
pixel=$(cat outputs_mnist_v3_standard_pixel.exit 2>/dev/null || echo running)
latent=$(cat outputs_mnist_v3_standard_latent.exit 2>/dev/null || echo waiting)
alive=$(ps -eo cmd | grep '[m]nist_fid_experiment.py' | wc -l)
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)
pixel_bytes=$(stat -c %s outputs_mnist_v3_standard_pixel.log 2>/dev/null || echo 0)
latent_bytes=$(stat -c %s outputs_mnist_v3_standard_latent.log 2>/dev/null || echo 0)
pixel_metrics=$(test -s outputs_mnist_v3_standard_pixel/metrics_summary.csv && echo yes || echo no)
latent_metrics=$(test -s outputs_mnist_v3_standard_latent/metrics_summary.csv && echo yes || echo no)
printf 'pixel=%s latent=%s alive=%s gpu="%s" pixel_log_bytes=%s latent_log_bytes=%s pixel_metrics=%s latent_metrics=%s\\n' \
  "$pixel" "$latent" "$alive" "$gpu" "$pixel_bytes" "$latent_bytes" "$pixel_metrics" "$latent_metrics"
"""


def parse_status(line: str) -> dict:
    fields = {}
    for part in line.strip().split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value.strip('"')
    return fields


def download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, item.filename)
        local_path = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            download_dir(sftp, remote_path, local_path)
        else:
            sftp.get(remote_path, str(local_path))


def fetch_outputs(client: paramiko.SSHClient) -> None:
    sftp = client.open_sftp()
    try:
        for name in [
            "outputs_mnist_v3_standard_pixel",
            "outputs_mnist_v3_standard_latent",
        ]:
            remote_path = posixpath.join(REMOTE_DIR, name)
            local_path = LOCAL_DIR / name
            try:
                if stat.S_ISDIR(sftp.stat(remote_path).st_mode):
                    download_dir(sftp, remote_path, local_path)
            except OSError:
                print(f"missing remote output dir: {remote_path}", flush=True)
        for name in [
            "outputs_mnist_v3_standard_pixel.log",
            "outputs_mnist_v3_standard_latent.log",
            "outputs_mnist_v3_standard_orchestrator.log",
            "outputs_mnist_v3_standard_pixel.exit",
            "outputs_mnist_v3_standard_latent.exit",
            "outputs_mnist_v3_standard.pid",
        ]:
            remote_path = posixpath.join(REMOTE_DIR, name)
            local_path = LOCAL_DIR / name
            try:
                sftp.get(remote_path, str(local_path))
            except OSError:
                pass
    finally:
        sftp.close()


def main() -> None:
    start = time.time()
    last_line = ""
    while True:
        if time.time() - start > MAX_WAIT_SECONDS:
            raise TimeoutError(f"remote standard experiment exceeded {MAX_WAIT_SECONDS} seconds")
        try:
            client = connect()
            try:
                code, out, err = capture(client, status_command())
                if code != 0:
                    raise RuntimeError(err or f"status command failed with exit code {code}")
                line = out.strip().splitlines()[-1] if out.strip() else ""
                if line and line != last_line:
                    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), line, flush=True)
                    last_line = line
                fields = parse_status(line)
                pixel = fields.get("pixel", "unknown")
                latent = fields.get("latent", "unknown")
                alive = fields.get("alive", "unknown")
                if pixel == "0" and latent == "0":
                    fetch_outputs(client)
                    print("FETCHED outputs_mnist_v3_standard_pixel and outputs_mnist_v3_standard_latent", flush=True)
                    return
                if pixel not in {"running", "0"} or latent not in {"waiting", "running", "0"}:
                    fetch_outputs(client)
                    raise RuntimeError(f"remote experiment failed: {line}")
                if alive == "0" and not (pixel == "0" and latent == "0"):
                    fetch_outputs(client)
                    raise RuntimeError(f"remote experiment stopped before completion: {line}")
            finally:
                client.close()
        except Exception as exc:
            print(time.strftime("[%Y-%m-%d %H:%M:%S]"), f"connect/status retry: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
