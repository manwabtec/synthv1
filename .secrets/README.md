# .secrets/: local credentials (never committed)

Everything in this folder except this README is ignored by git (`.gitignore`) and
excluded from the Docker build context (`infra/docker/Dockerfile.dockerignore`), so it
never reaches GitHub or an image. Keep it private: `chmod 700 .secrets`, `chmod 600` files.

Use it for **git push credentials only**. AWS access uses SSO or the instance role; never
store AWS keys here or anywhere else.

## Option A: HTTPS + personal access token (simplest)

Create a fine-grained token on GitHub (Settings → Developer settings → Fine-grained
tokens) with access to this repository only and "Contents: Read and write". Then, from
the repository root:

```bash
git config credential.helper "store --file=$PWD/.secrets/git-credentials"
git push            # username = your GitHub user, password = the token; saved once
chmod 600 .secrets/git-credentials
```

`git config` without `--global` applies to this clone only. The token is stored in
plain text in `.secrets/git-credentials`, so give it an expiry date.

## Option B: SSH deploy key (per machine, this repository only)

```bash
ssh-keygen -t ed25519 -N '' -f .secrets/github_deploy_key -C "synthrig-$(hostname)"
cat .secrets/github_deploy_key.pub
#   GitHub: repo Settings → Deploy keys → Add key, tick "Allow write access"
git config core.sshCommand "ssh -i $PWD/.secrets/github_deploy_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git remote set-url origin git@github.com:manwabtec/synthv1.git
git push
```

To revoke access, delete the token or deploy key on GitHub and remove the file here.
