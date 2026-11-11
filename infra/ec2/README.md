# EC2 instance scripts

| File | Runs on | Purpose |
|---|---|---|
| `setup_host.sh` | EC2 (sudo) | Creates `/app/synthrig` on EBS, mounts the instance-store NVMe at `/scratch` (it is erased on every stop/start, so re-run then), registers the NVIDIA Docker runtime, moves Docker's data-root to `/app/docker`, and writes `infra/config.yaml` |
| `check_ec2.sh` | EC2 | End-to-end check: system, **disk space**, GPU, GPU in Docker, image build + test, gpucheck (GPU vs CPU), a demo render with preview, and `cloud.py status`. `--quick` skips the renders; `--uv` adds the host uv/.venv checks |

Logs go to `<data_root>/logs/ec2/<UTC-time>_<commit>/`: `summary.txt` (read this first),
`run_info.txt`, `NN_<step>.log` and `preview_app1.png`. Nothing is written into the git
checkout. Neither script starts AWS resources or prints credentials.

## One-time setup on the instance

Work on the instance through VS Code (Remote) or an SSM session.

```bash
sudo dnf install -y git tmux                              # if missing
git clone -b InitialCommit https://github.com/manwabtec/synthv1.git code && cd code
# push access: see .secrets/README.md (token or deploy key, stored in .secrets/)
bash infra/ec2/setup_host.sh
```

## Each round

Use `tmux` so a dropped session doesn't kill a long run (`tmux attach` resumes it).

```bash
tmux new -s check                       # or: tmux attach -t check
bash infra/ec2/setup_host.sh            # only after a stop/start (re-mounts /scratch)
bash infra/ec2/check_ec2.sh --pull      # add --quick to skip the renders
cat /app/synthrig/logs/ec2/$(ls -1 /app/synthrig/logs/ec2 | tail -1)/summary.txt
```
