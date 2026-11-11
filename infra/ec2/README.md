# EC2 instance scripts

| File | Runs on | Purpose |
|---|---|---|
| `setup_host.sh` | EC2 (sudo) | Creates `/app/synthrig` on EBS, mounts the instance-store NVMe at `/scratch` (it is erased on every stop/start, so re-run then), registers the NVIDIA Docker runtime, moves Docker's data-root to `/app/docker`, and writes `infra/config.yaml` |

The end-to-end check is `infra/check.sh` (see `infra/README.md`); it works on EC2 and on a
local NVIDIA machine. `setup_host.sh` never starts AWS resources or prints credentials.

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
bash infra/check.sh --pull              # add --quick to skip the renders
cat /app/synthrig/logs/check/$(ls -1 /app/synthrig/logs/check | tail -1)/summary.txt
```
