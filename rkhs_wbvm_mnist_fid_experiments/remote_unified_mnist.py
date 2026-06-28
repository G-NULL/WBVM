import argparse
import posixpath
import shlex
import stat
import time
from pathlib import Path

import paramiko

from remote_vector_mmd import capture, connect, install_deps, upload_files, upload_mnist_raw


LOCAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTDIR = "outputs_mnist_unified_dit_mnistfid"
DEFAULT_METHODS = "wbvm_single,wbvm_vector,meanflow,shortcut,drifting"


def q(value: str) -> str:
    return shlex.quote(str(value))


def run_script_text(args: argparse.Namespace) -> str:
    parts = [
        q(args.python_bin),
        "mnist_fid_experiment.py",
        "--preset",
        q(args.preset),
        "--model-space",
        "pixel",
        "--methods",
        q(args.methods),
        "--direct-backbone",
        q(args.direct_backbone),
        "--fid-backend",
        "mnist",
        "--selection-metric",
        "mnist_fid",
        "--vector-loss-statistic",
        q(args.vector_loss_statistic),
        "--num-threads",
        str(args.num_threads),
        "--num-workers",
        str(args.num_workers),
        "--outdir",
        q(args.outdir_name),
    ]
    if args.tune_baselines:
        parts.append("--tune-baselines")
    if args.early_stop_min_steps is not None:
        parts.extend(["--early-stop-min-steps", str(args.early_stop_min_steps)])
    if args.early_stop_patience is not None:
        parts.extend(["--early-stop-patience", str(args.early_stop_patience)])
    if args.early_stop_min_delta is not None:
        parts.extend(["--early-stop-min-delta", str(args.early_stop_min_delta)])
    if args.early_stop_metric is not None:
        parts.extend(["--early-stop-metric", q(args.early_stop_metric)])
    if args.steps is not None:
        parts.extend(["--steps", str(args.steps)])
    if args.single_steps is not None:
        parts.extend(["--single-steps", str(args.single_steps)])
    if args.wbvm_single_taus is not None:
        parts.extend(["--wbvm-single-taus", q(args.wbvm_single_taus)])
    if args.train_n is not None:
        parts.extend(["--train-n", str(args.train_n)])
    if args.fid_samples is not None:
        parts.extend(["--fid-samples", str(args.fid_samples)])
    if args.batch_size is not None:
        parts.extend(["--batch-size", str(args.batch_size)])
    if args.kernel_batch is not None:
        parts.extend(["--kernel-batch", str(args.kernel_batch)])
    command = " ".join(parts)
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd {q(args.remote_dir)}
echo $$ > {q(args.outdir_name + ".pid")}
echo running > {q(args.outdir_name + ".exit")}

{command} > {q(args.outdir_name + ".log")} 2>&1
status=$?
echo "$status" > {q(args.outdir_name + ".exit")}
exit "$status"
"""


def write_remote_run_script(client: paramiko.SSHClient, args: argparse.Namespace) -> str:
    remote_script = posixpath.join(args.remote_dir, f"run_{args.outdir_name}.sh")
    script = run_script_text(args)
    sftp = client.open_sftp()
    try:
        with sftp.file(remote_script, "w") as f:
            f.write(script)
        sftp.chmod(remote_script, 0o755)
    finally:
        sftp.close()
    return remote_script


def start_run(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    remote_script = write_remote_run_script(client, args)
    if args.clean_remote_output:
        cleanup = (
            f"cd {q(args.remote_dir)}; "
            f"rm -rf {q(args.outdir_name)} "
            f"{q(args.outdir_name + '.pid')} {q(args.outdir_name + '.exit')} "
            f"{q(args.outdir_name + '.log')} {q(args.outdir_name + '_orchestrator.log')}"
        )
        code, _, err = capture(client, cleanup, timeout=60)
        if code != 0:
            raise RuntimeError(err or "failed to clean previous unified MNIST output")
    command = (
        f"cd {q(args.remote_dir)}; "
        f"nohup bash {q(remote_script)} "
        f"> {q(args.outdir_name + '_orchestrator.log')} 2>&1 < /dev/null & "
        "echo $!"
    )
    code, out, err = capture(client, command, timeout=60)
    if code != 0:
        raise RuntimeError(err or "failed to start unified MNIST run")
    print(f"STARTED unified MNIST run pid={out.strip()} outdir={args.outdir_name}", flush=True)


def status_command(args: argparse.Namespace) -> str:
    out = args.outdir_name
    return f"""
cd {q(args.remote_dir)}
status=$(cat {q(out + ".exit")} 2>/dev/null || echo running)
alive=$(ps -eo cmd | grep -E '[m]nist_fid_experiment.py.*{out}|[r]un_{out}.sh' | wc -l)
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1 2>/dev/null || echo no-gpu)
log_bytes=$(stat -c %s {q(out + ".log")} 2>/dev/null || echo 0)
metrics=$(test -s {q(out + "/metrics_summary.csv")} && echo yes || echo no)
samples=$(test -s {q(out + "/sample_grid.pt")} && echo yes || echo no)
printf 'status=%s alive=%s gpu="%s" log_bytes=%s metrics=%s samples=%s\\n' \\
  "$status" "$alive" "$gpu" "$log_bytes" "$metrics" "$samples"
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
            raise TimeoutError(f"remote unified MNIST run exceeded {args.max_wait_seconds} seconds")
        code, out, err = capture(client, status_command(args), timeout=60)
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
            fetch_outputs(client, args)
            raise RuntimeError(f"remote unified MNIST run failed: {line}")
        if alive == "0":
            fetch_outputs(client, args)
            raise RuntimeError(f"remote unified MNIST run stopped before completion: {line}")
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


def fetch_outputs(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    sftp = client.open_sftp()
    try:
        remote_path = posixpath.join(args.remote_dir, args.outdir_name)
        local_path = LOCAL_DIR / args.outdir_name
        try:
            if stat.S_ISDIR(sftp.stat(remote_path).st_mode):
                download_dir(sftp, remote_path, local_path)
        except OSError:
            print(f"missing remote output dir: {remote_path}", flush=True)
        for name in [
            f"{args.outdir_name}.pid",
            f"{args.outdir_name}.exit",
            f"{args.outdir_name}.log",
            f"{args.outdir_name}_orchestrator.log",
            f"run_{args.outdir_name}.sh",
        ]:
            try:
                sftp.get(posixpath.join(args.remote_dir, name), str(LOCAL_DIR / name))
            except OSError:
                pass
    finally:
        sftp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified DriftDiT-style MNIST-FID experiments remotely")
    parser.add_argument("--host", default="connect.westb.seetacloud.com")
    parser.add_argument("--port", type=int, default=23522)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=None)
    parser.add_argument("--remote-dir", default="/root/rkhs_wbvm_mnist_fid_experiments")
    parser.add_argument("--outdir-name", default=DEFAULT_OUTDIR)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--preset", default="standard", choices=["smoke", "quick", "standard"])
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python3")
    parser.add_argument("--direct-backbone", default="dit", choices=["dit", "cnn"])
    parser.add_argument("--vector-loss-statistic", default="u", choices=["u", "v"])
    parser.add_argument("--single-steps", type=int, default=None)
    parser.add_argument("--wbvm-single-taus", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--train-n", type=int, default=None)
    parser.add_argument("--fid-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--kernel-batch", type=int, default=None)
    parser.add_argument("--tune-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-min-steps", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--early-stop-metric", choices=["loss", "abs_loss"], default=None)
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--upload-mnist-raw", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--clean-remote-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--status-once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=12 * 60 * 60)
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
        if args.install_deps:
            install_deps(client, args.remote_dir)
        if args.status_once:
            code, out, err = capture(client, status_command(args), timeout=60)
            if code != 0:
                raise RuntimeError(err or f"status command failed with exit code {code}")
            print(out.strip(), flush=True)
            return
        if not args.no_start:
            start_run(client, args)
        if not args.no_wait:
            wait_for_run(client, args)
        if not args.no_fetch:
            fetch_outputs(client, args)
            print("FETCHED unified MNIST outputs", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
