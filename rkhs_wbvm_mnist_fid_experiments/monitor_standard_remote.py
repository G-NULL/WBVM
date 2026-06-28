import os
import time

import paramiko


HOST = "connect.westb.seetacloud.com"
PORT = 47830
USER = "root"
REMOTE_DIR = "/root/rkhs_wbvm_mnist_fid_experiments"


def capture(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), output, error


def main() -> None:
    password = os.environ["SEETA_PASS"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=password, timeout=30)
    try:
        while True:
            command = f"""
cd {REMOTE_DIR}
pixel=$(cat outputs_mnist_v3_standard_pixel.exit 2>/dev/null || echo running)
latent=$(cat outputs_mnist_v3_standard_latent.exit 2>/dev/null || echo waiting)
alive=$(ps -eo cmd | grep '[m]nist_fid_experiment.py' | wc -l)
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)
pixel_bytes=$(stat -c %s outputs_mnist_v3_standard_pixel.log 2>/dev/null || echo 0)
latent_bytes=$(stat -c %s outputs_mnist_v3_standard_latent.log 2>/dev/null || echo 0)
printf 'pixel=%s latent=%s alive=%s gpu=%s pixel_log_bytes=%s latent_log_bytes=%s\n' \
  "$pixel" "$latent" "$alive" "$gpu" "$pixel_bytes" "$latent_bytes"
"""
            code, output, error = capture(client, command)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{stamp}] {output.strip()}", flush=True)
            if code != 0:
                raise RuntimeError(error or f"monitor command failed with exit code {code}")
            fields = output.splitlines()[0] if output else ""
            if "pixel=0 latent=0" in fields:
                return
            if "alive=0" in fields and ("pixel=running" in fields or "latent=waiting" in fields):
                raise RuntimeError("Training process stopped without writing both exit-status files")
            time.sleep(60)
    finally:
        client.close()


if __name__ == "__main__":
    main()
