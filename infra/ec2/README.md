# EC2 pipeline check

Use this when you can't develop on the instance directly. Edit and push on the laptop,
then pull and run on EC2, then push the log back and read it on the laptop.

| File | Runs on | Purpose |
|---|---|---|
| `setup_host.sh` | EC2 (sudo) | Creates `/app/synthrig` on EBS, mounts the instance-store NVMe at `/scratch` (re-run after every stop/start: it is erased), registers the NVIDIA Docker runtime and moves Docker's data-root to `/app/docker` |
| `check_ec2.sh` | EC2 | Checks the system, **disk space**, GPU, GPU-in-Docker, image build + test, gpucheck (GPU vs CPU), a demo render with preview, and `cloud.py status`, then writes a log folder. `--uv` adds the host uv/.venv checks; `--quick` skips the renders |
| `push_log.sh` | EC2 | Commits one log folder and pushes it |
| `../../logs/ec2/<UTC-time>_<commit>/` | both | One folder per run: `summary.txt`, `run_info.txt`, `NN_<step>.log`, `preview_app1.png` |

None of these scripts starts AWS resources or prints credentials.

## One-time setup on the instance

Open a shell with `./aws_connection_check/ssh_ec2.sh` (from the laptop's parent folder), then:

```bash
sudo su - ec2-user                 # Session Manager logs in as ssm-user
sudo dnf install -y git tmux       # if missing

# Read+write GitHub access for this instance only (deploy key):
ssh-keygen -t ed25519 -N '' -f ~/.ssh/github_synthv1 -C synthrig-ec2
cat ~/.ssh/github_synthv1.pub
#   -> GitHub: repo Settings > Deploy keys > Add key, tick "Allow write access"
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile ~/.ssh/github_synthv1
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh -T git@github.com              # expect "successfully authenticated"

mkdir -p ~/synthv1 && cd ~/synthv1
git clone -b InitialCommit git@github.com:manwabtec/synthv1.git code
```

The default checks use Docker and the system `python3`; `uv` is only needed with `--uv`
(add `--install-uv` to install it user-level).

## Each round

Run on the instance. Use `tmux` so a dropped session doesn't kill the run;
`tmux attach` resumes it.

```bash
cd ~/synthv1/code
tmux new -s check                       # or: tmux attach -t check
bash infra/ec2/setup_host.sh            # only after a stop/start (re-mounts /scratch)
bash infra/ec2/check_ec2.sh --pull      # add --quick to skip the renders
bash infra/ec2/push_log.sh
```

Then on the laptop, run `git pull` and open `logs/ec2/<newest>/summary.txt`.
