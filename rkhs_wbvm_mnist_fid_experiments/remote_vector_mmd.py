import argparse
import posixpath
import stat
import time
from pathlib import Path
from typing import Iterable, Tuple

import paramiko


LOCAL_DIR = Path(__file__).resolve().parent


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def capture(client: paramiko.SSHClient, command: str, timeout: int = 60) -> Tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def files_to_upload() -> Iterable[Path]:
    for path in sorted(LOCAL_DIR.iterdir()):
        if path.is_file() and (path.suffix in {".py", ".sh", ".md"} or path.name == "requirements.txt"):
            yield path


def upload_files(client: paramiko.SSHClient, remote_dir: str) -> None:
    code, _, err = capture(client, f"mkdir -p {remote_dir}")
    if code != 0:
        raise RuntimeError(err or f"failed to create {remote_dir}")
    sftp = client.open_sftp()
    try:
        for local_path in files_to_upload():
            remote_path = posixpath.join(remote_dir, local_path.name)
            sftp.put(str(local_path), remote_path)
        for name in [
            "run_vector_mmd_both.sh",
            "run_vector_mmd_pixel_cnn.sh",
            "run_vector_mmd_pixel_dit.sh",
            "run_mnist_fid_cnn_all.sh",
        ]:
            try:
                sftp.chmod(posixpath.join(remote_dir, name), 0o755)
            except OSError:
                pass
    finally:
        sftp.close()


def upload_mnist_raw(client: paramiko.SSHClient, remote_dir: str) -> None:
    local_raw = LOCAL_DIR / "data" / "MNIST" / "raw"
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    missing = [name for name in required if not (local_raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local MNIST raw files; run local torchvision MNIST once: {missing}")
    files = [path for path in sorted(local_raw.iterdir()) if path.is_file()]
    remote_raw = posixpath.join(remote_dir, "data", "MNIST", "raw")
    code, _, err = capture(client, f"mkdir -p {remote_raw}")
    if code != 0:
        raise RuntimeError(err or f"failed to create {remote_raw}")
    sftp = client.open_sftp()
    try:
        for local_path in files:
            name = local_path.name
            remote_path = posixpath.join(remote_raw, name)
            sftp.put(str(local_path), remote_path)
            print(f"UPLOADED {name} ({local_path.stat().st_size} bytes)", flush=True)
    finally:
        sftp.close()


def install_deps(client: paramiko.SSHClient, remote_dir: str) -> None:
    packages = "scipy torchmetrics torch-fidelity tqdm matplotlib pandas"
    command = f"cd {remote_dir} && /root/miniconda3/bin/python3 -m pip install {packages}"
    code, out, err = capture(client, command, timeout=900)
    if out.strip():
        print(out.strip(), flush=True)
    if err.strip():
        print(err.strip(), flush=True)
    if code != 0:
        raise RuntimeError(f"dependency installation failed with exit code {code}")


def stop_current_run(client: paramiko.SSHClient, remote_dir: str) -> None:
    command = f"""
pkill -f '[m]nist_fid_experiment.py.*outputs_mnist_route2_cnn_pixel' 2>/dev/null || true
pkill -f '[r]un_vector_mmd_pixel_cnn.sh' 2>/dev/null || true
cd {remote_dir}
echo stopped > outputs_mnist_route2_cnn_pixel.exit
"""
    code, out, err = capture(client, command, timeout=60)
    if out.strip():
        print(out.strip(), flush=True)
    if err.strip():
        print(err.strip(), flush=True)
    if code != 0:
        raise RuntimeError(f"failed to stop current route-two run with exit code {code}")


def start_run(client: paramiko.SSHClient, remote_dir: str) -> None:
    command = (
        f"cd {remote_dir}; "
        "nohup bash run_vector_mmd_pixel_cnn.sh "
        "> outputs_mnist_route2_cnn_pixel_orchestrator.log 2>&1 < /dev/null & "
        "echo $!"
    )
    code, out, err = capture(client, command)
    if code != 0:
        raise RuntimeError(err or "failed to start route-two MNIST run")
    print(f"STARTED remote route-two MNIST run pid={out.strip()}", flush=True)


def status_command(remote_dir: str) -> str:
    return f"""
cd {remote_dir}
pixel=$(cat outputs_mnist_route2_cnn_pixel.exit 2>/dev/null || echo running)
alive=$(ps -eo cmd | grep -E '[m]nist_fid_experiment.py|[r]un_vector_mmd_pixel_cnn.sh' | wc -l)
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1 2>/dev/null || echo no-gpu)
pixel_bytes=$(stat -c %s outputs_mnist_route2_cnn_pixel.log 2>/dev/null || echo 0)
pixel_metrics=$(test -s outputs_mnist_route2_cnn_pixel/metrics_summary.csv && echo yes || echo no)
printf 'pixel=%s alive=%s gpu="%s" pixel_log_bytes=%s pixel_metrics=%s\\n' \
  "$pixel" "$alive" "$gpu" "$pixel_bytes" "$pixel_metrics"
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
            raise TimeoutError(f"remote route-two MNIST run exceeded {args.max_wait_seconds} seconds")
        code, out, err = capture(client, status_command(args.remote_dir), timeout=60)
        if code != 0:
            raise RuntimeError(err or f"status command failed with exit code {code}")
        line = out.strip().splitlines()[-1] if out.strip() else ""
        if line and line != last_line:
            print(time.strftime("[%Y-%m-%d %H:%M:%S]"), line, flush=True)
            last_line = line
        fields = parse_status(line)
        pixel = fields.get("pixel", "unknown")
        alive = fields.get("alive", "unknown")
        if pixel == "0":
            return
        if pixel != "running":
            raise RuntimeError(f"remote experiment failed: {line}")
        if alive == "0" and pixel != "0":
            raise RuntimeError(f"remote experiment stopped before completion: {line}")
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
        for name in [
            "outputs_mnist_route2_cnn_pixel",
        ]:
            remote_path = posixpath.join(remote_dir, name)
            local_path = LOCAL_DIR / name
            try:
                if stat.S_ISDIR(sftp.stat(remote_path).st_mode):
                    download_dir(sftp, remote_path, local_path)
            except OSError:
                print(f"missing remote output dir: {remote_path}", flush=True)
        for name in [
            "outputs_mnist_route2_cnn_pixel.pid",
            "outputs_mnist_route2_cnn_pixel.exit",
            "outputs_mnist_route2_cnn_pixel.log",
            "outputs_mnist_route2_cnn_pixel_orchestrator.log",
        ]:
            remote_path = posixpath.join(remote_dir, name)
            local_path = LOCAL_DIR / name
            try:
                sftp.get(remote_path, str(local_path))
            except OSError:
                pass
    finally:
        sftp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MNIST route-two vMMD-WBVM on the remote server")
    parser.add_argument("--host", default="connect.westd.seetacloud.com")
    parser.add_argument("--port", type=int, default=22345)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=None)
    parser.add_argument("--remote-dir", default="/root/rkhs_wbvm_mnist_fid_experiments")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--upload-mnist-raw", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--status-once", action="store_true")
    parser.add_argument("--stop-current", action="store_true")
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
            print("UPLOADED MNIST route-two files", flush=True)
        if args.upload_mnist_raw:
            upload_mnist_raw(client, args.remote_dir)
        if args.install_deps:
            install_deps(client, args.remote_dir)
        if args.stop_current:
            stop_current_run(client, args.remote_dir)
            print("STOPPED current route-two MNIST run", flush=True)
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
            print("FETCHED route-two MNIST outputs", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
