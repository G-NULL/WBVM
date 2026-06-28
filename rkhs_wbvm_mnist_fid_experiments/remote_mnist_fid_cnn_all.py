import argparse
import posixpath
import stat
import time
from pathlib import Path

import paramiko

from remote_vector_mmd import capture, connect, upload_files, upload_mnist_raw


LOCAL_DIR = Path(__file__).resolve().parent
OUTDIR = "outputs_mnist_corrected_cnn_mnistfid"


def start_run(client: paramiko.SSHClient, remote_dir: str) -> None:
    command = (
        f"cd {remote_dir}; "
        "nohup bash run_mnist_fid_cnn_all.sh "
        "> outputs_mnist_corrected_cnn_mnistfid_orchestrator.log 2>&1 < /dev/null & "
        "echo $!"
    )
    code, out, err = capture(client, command, timeout=60)
    if code != 0:
        raise RuntimeError(err or "failed to start all-model MNIST-FID run")
    print(f"STARTED all-model MNIST-FID run pid={out.strip()}", flush=True)


def status_command(remote_dir: str) -> str:
    return f"""
cd {remote_dir}
status=$(cat outputs_mnist_corrected_cnn_mnistfid.exit 2>/dev/null || echo running)
alive=$(ps -eo cmd | grep -E '[m]nist_fid_experiment.py|[r]un_mnist_fid_cnn_all.sh' | wc -l)
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1 2>/dev/null || echo no-gpu)
log_bytes=$(stat -c %s outputs_mnist_corrected_cnn_mnistfid.log 2>/dev/null || echo 0)
metrics=$(test -s outputs_mnist_corrected_cnn_mnistfid/metrics_summary.csv && echo yes || echo no)
printf 'status=%s alive=%s gpu="%s" log_bytes=%s metrics=%s\\n' \
  "$status" "$alive" "$gpu" "$log_bytes" "$metrics"
"""


def parse_status(line: str) -> dict:
    fields = {}
    current = ""
    in_quote = False
    parts = []
    for char in line.strip():
        if char == '"':
            in_quote = not in_quote
            current += char
        elif char == " " and not in_quote:
            if current:
                parts.append(current)
                current = ""
        else:
            current += char
    if current:
        parts.append(current)
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value.strip('"')
    return fields


def wait_for_run(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    start = time.time()
    last_line = ""
    while True:
        if time.time() - start > args.max_wait_seconds:
            raise TimeoutError(f"remote MNIST-FID run exceeded {args.max_wait_seconds} seconds")
        code, out, err = capture(client, status_command(args.remote_dir), timeout=60)
        if code != 0:
            raise RuntimeError(err or f"status command failed with exit code {code}")
        line = out.strip().splitlines()[-1] if out.strip() else ""
        if line and line != last_line:
            print(time.strftime("[%Y-%m-%d %H:%M:%S]"), line, flush=True)
            last_line = line
        fields = parse_status(line)
        status = fields.get("status", "unknown")
        alive = fields.get("alive", "unknown")
        if status == "0":
            return
        if status != "running":
            raise RuntimeError(f"remote MNIST-FID run failed: {line}")
        if alive == "0":
            raise RuntimeError(f"remote MNIST-FID run stopped before completion: {line}")
        time.sleep(args.poll_seconds)


def download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, item.filename)
        local_path = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            download_dir(sftp, remote_path, local_path)
        else:
            sftp.get(remote_path, str(local_path))


def fetch_outputs(client: paramiko.SSHClient, remote_dir: str) -> None:
    sftp = client.open_sftp()
    try:
        remote_path = posixpath.join(remote_dir, OUTDIR)
        local_path = LOCAL_DIR / OUTDIR
        if stat.S_ISDIR(sftp.stat(remote_path).st_mode):
            download_dir(sftp, remote_path, local_path)
        for name in [
            f"{OUTDIR}.pid",
            f"{OUTDIR}.exit",
            f"{OUTDIR}.log",
            f"{OUTDIR}_orchestrator.log",
        ]:
            try:
                sftp.get(posixpath.join(remote_dir, name), str(LOCAL_DIR / name))
            except OSError:
                pass
    finally:
        sftp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pixel-CNN MNIST-FID for corrected-standard methods")
    parser.add_argument("--host", default="connect.westd.seetacloud.com")
    parser.add_argument("--port", type=int, default=22345)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=None)
    parser.add_argument("--remote-dir", default="/root/rkhs_wbvm_mnist_fid_experiments")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--upload-mnist-raw", action="store_true")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--status-once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=8 * 60 * 60)
    args = parser.parse_args()
    if args.password is None:
        import os

        args.password = os.environ["SEETA_PASS"]

    client = connect(args)
    try:
        if not args.no_upload:
            upload_files(client, args.remote_dir)
            print("UPLOADED MNIST experiment files", flush=True)
        if args.upload_mnist_raw:
            upload_mnist_raw(client, args.remote_dir)
        if args.status_once:
            code, out, err = capture(client, status_command(args.remote_dir), timeout=60)
            if code != 0:
                raise RuntimeError(err or f"status command failed with exit code {code}")
            print(out.strip(), flush=True)
            return
        if not args.no_start:
            start_run(client, args.remote_dir)
        if not args.no_wait:
            wait_for_run(client, args)
        if not args.no_fetch:
            fetch_outputs(client, args.remote_dir)
            print("FETCHED all-model MNIST-FID outputs", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
