import argparse
import os
import posixpath
import shlex
import stat
import sys
import time
from pathlib import Path

import paramiko


def connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    cur = "/" if remote_dir.startswith("/") else "."
    for part in parts:
        cur = posixpath.join(cur, part)
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def upload_dir(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    sftp_mkdirs(sftp, remote_dir)
    for root, dirs, files in os.walk(local_dir):
        root_path = Path(root)
        excluded_dirs = {"__pycache__", "weights_cache", ".venv", ".git", "data"}
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith("outputs")]
        rel = root_path.relative_to(local_dir).as_posix()
        rroot = remote_dir if rel == "." else posixpath.join(remote_dir, rel)
        sftp_mkdirs(sftp, rroot)
        for d in dirs:
            sftp_mkdirs(sftp, posixpath.join(rroot, d))
        for f in files:
            if "__pycache__" in root_path.parts or f.endswith((".pyc", ".pyo")):
                continue
            sftp.put(str(root_path / f), posixpath.join(rroot, f))


def is_dir(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(remote_path).st_mode)
    except IOError:
        return False


def download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        rpath = posixpath.join(remote_dir, item.filename)
        lpath = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            download_dir(sftp, rpath, lpath)
        else:
            sftp.get(rpath, str(lpath))


def run_stream(client: paramiko.SSHClient, command: str) -> int:
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("No SSH transport")
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command(command)
    def write_stream(stream, text: str) -> None:
        try:
            stream.write(text)
            stream.flush()
        except UnicodeEncodeError:
            stream.buffer.write(text.encode("utf-8", errors="replace"))
            stream.buffer.flush()

    while True:
        while chan.recv_ready():
            write_stream(sys.stdout, chan.recv(4096).decode("utf-8", errors="replace"))
        while chan.recv_stderr_ready():
            write_stream(sys.stderr, chan.recv_stderr(4096).decode("utf-8", errors="replace"))
        if chan.exit_status_ready():
            break
        time.sleep(0.2)
    return chan.recv_exit_status()


def run_capture(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def detect_remote_python(client: paramiko.SSHClient) -> str:
    candidates = [
        "python3",
        "python",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/conda/bin/python",
        "/root/miniconda3/bin/python",
        "/root/anaconda3/bin/python",
        "/root/miniconda/bin/python",
    ]
    script = " ; ".join(
        [f"command -v {p} >/dev/null 2>&1 && command -v {p} && exit 0" for p in candidates if not p.startswith("/")]
        + [f"[ -x {p} ] && echo {p} && exit 0" for p in candidates if p.startswith("/")]
        + ["exit 1"]
    )
    code, out, err = run_capture(client, f"bash -lc '{script}'")
    if code != 0:
        raise RuntimeError(f"Could not find a Python interpreter on remote host.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out.strip().splitlines()[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload/run/fetch RKHS-WBVM experiment over SSH")
    parser.add_argument("--host", default="connect.westb.seetacloud.com")
    parser.add_argument("--port", type=int, default=10627)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=os.environ.get("SEETA_PASS", ""))
    parser.add_argument("--remote-dir", default="/root/rkhs_wbvm_synth_experiments")
    parser.add_argument("--local-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--outdir-name", default="outputs_remote")
    parser.add_argument("--preset", default="quick", choices=["smoke", "quick", "standard"])
    parser.add_argument("--remote-python", default="auto")
    parser.add_argument("--command", default=None, help="Run this command after upload instead of the experiment command.")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-vector-mmd", action="store_true", help="Run the route-two vector-flux MMD WBVM experiment.")
    parser.add_argument("--background", action="store_true", help="Start --run-vector-mmd in the background and return after writing a pid file.")
    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--all-tau-low", default=None)
    parser.add_argument("--all-tau-high", default=None)
    parser.add_argument("--val-metric", default=None, choices=[None, "flux", "sliced_w2", "w2", "energy"])
    parser.add_argument("--loss-statistic", default=None, choices=[None, "v", "u"])
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--skip-knt", action="store_true")
    args = parser.parse_args()

    if not args.password:
        raise SystemExit("Missing password. Pass --password or set SEETA_PASS.")

    local_dir = Path(args.local_dir).resolve()
    client = connect(args.host, args.port, args.user, args.password)
    sftp = client.open_sftp()
    if not args.no_upload:
        print(f"Uploading {local_dir} -> {args.remote_dir}")
        upload_dir(sftp, local_dir, args.remote_dir)

    remote_python = detect_remote_python(client) if args.remote_python == "auto" else args.remote_python
    print(f"Remote Python: {remote_python}")

    if args.command:
        code = run_stream(client, args.command)
        if code != 0:
            raise SystemExit(f"Remote command failed with exit code {code}")

    if args.run_vector_mmd:
        env = {
            "PYTHON_BIN": remote_python,
            "OUTDIR": args.outdir_name,
            "PRESET": args.preset,
            "TRAIN_N": str(args.train_n),
            "THREADS": str(args.num_threads),
        }
        if args.all_tau_low is not None:
            env["ALL_TAU_LOW"] = str(args.all_tau_low)
        if args.all_tau_high is not None:
            env["ALL_TAU_HIGH"] = str(args.all_tau_high)
        if args.val_metric is not None:
            env["VAL_METRIC"] = args.val_metric
        if args.loss_statistic is not None:
            env["LOSS_STATISTIC"] = args.loss_statistic
        env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        if args.background:
            pid_file = "vector_mmd_2602sec61.pid"
            exit_file = "vector_mmd_2602sec61.exit"
            log_file = "vector_mmd_2602sec61.log"
            command = (
                f"cd {shlex.quote(args.remote_dir)} && "
                f"rm -f {pid_file} {exit_file} {log_file} && "
                f"({env_text} EXIT_FILE={exit_file} "
                f"nohup bash run_vector_mmd_remote.sh > {log_file} 2>&1 & "
                f"echo $! > {pid_file}) && "
                f"echo started $(cat {pid_file})"
            )
        else:
            command = f"cd {shlex.quote(args.remote_dir)} && {env_text} bash run_vector_mmd_remote.sh"
        code = run_stream(client, command)
        if code != 0:
            raise SystemExit(f"Remote command failed with exit code {code}")

    if args.run:
        skip = " --skip-knt" if args.skip_knt else ""
        command = (
            f"cd {args.remote_dir} && "
            f"({remote_python} - <<'PY'\n"
            f"import importlib.util\n"
            f"missing=[p for p in ['numpy','matplotlib','torch','scipy'] if importlib.util.find_spec(p) is None]\n"
            f"print('missing', missing)\n"
            f"PY\n"
            f") && "
            f"{remote_python} wbvm_rkhs_experiment.py --preset {args.preset} --outdir {args.outdir_name}{skip}"
        )
        code = run_stream(client, command)
        if code != 0:
            raise SystemExit(f"Remote command failed with exit code {code}")

    if args.fetch:
        remote_out = posixpath.join(args.remote_dir, args.outdir_name)
        local_out = local_dir / args.outdir_name
        if is_dir(sftp, remote_out):
            print(f"Fetching {remote_out} -> {local_out}")
            download_dir(sftp, remote_out, local_out)
        else:
            print(f"Remote output directory does not exist yet: {remote_out}")

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
