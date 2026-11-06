# AWS infrastructure

This directory will implement Phase 0 from `docs/TASK_infra.md`. It is safe to
prepare locally: no AWS resource is created until `services start --yes` is run.

## AWS information required

Ask the AWS administrator for:

- the AWS account ID and an AWS CLI v2 SSO profile with permission to use it;
- the approved region and GPU instance families, plus EC2 and Spot quotas;
- an owner/budget-alert email, cost centre and mandatory resource tags;
- a globally unique S3 bucket name and the organisation's required encryption
  policy (SSE-S3 or a KMS key ARN);
- whether a new VPC is allowed; otherwise, the VPC ID and private subnet IDs;
- any IAM permission-boundary ARN, service-control policies and required roles;
- approved Deep Learning/ECS GPU AMI owners or an approved AMI selection rule;
- permission to use ECR, S3, EC2, Batch, IAM, SSM, CloudWatch, Budgets and the
  required VPC endpoints;
- the desired monthly budget and alert recipients.

Do not request access keys. Run `aws configure sso`, and place only the SSO
profile name in the local config. Account identity is discovered through STS.

## Local configuration

Copy `infra/config.example.yaml` to `infra/config.yaml` and fill it in. The
local file is gitignored. It is the only account-specific file in this repo;
Terraform variables are generated from it into another ignored file.

```powershell
Copy-Item infra/config.example.yaml infra/config.yaml
uv sync
uv run python infra/cloud.py config
```

## Lifecycle

A lightweight laptop can run configuration tools, tests and downloaded-dataset
evaluation. A separate NVIDIA workstation can also render locally; run
`uv sync --extra render`, then `uv run python infra/cloud.py local-check --gpu --render`.
The EC2 and Batch render commands remain unavailable until their Phase 0 steps
are implemented. Local verification does not complete the AWS acceptance checks.

`access/check.py` checks AWS identity, EC2 read access and SSM connectivity:
`uv run python -m infra.access.check <instance-id>`. `diagnostics/` checks local
execution only and never requires AWS credentials. `storage/s3.py` implements
snapshot push/pull, exposed through `cloud.py`; see the root README for commands,
checksum semantics and immutable-prefix requirements. AWS data persists in S3;
workers will use local scratch, not the workstation data root, while rendering.

The Terraform design has two states:

- `storage` owns only the private S3 bucket and its protection/lifecycle rules;
- `runtime` owns ECR, networking/endpoints, IAM, logs, the dev instance and
  Batch resources.

Once Phase 0.2's Terraform stacks exist, these are the complete start and stop
scripts:

```powershell
uv run python infra/cloud.py services start --yes
uv run python infra/cloud.py services status
uv run python infra/cloud.py services stop --yes
```

`stop` destroys the runtime state instead of merely stopping EC2. This matters
because stopped-instance EBS volumes, ECR images, CloudWatch logs, NAT gateways
and interface VPC endpoints can still cost money. It preserves the S3 bucket,
including Terraform state and datasets. S3 storage, requests, retrieval, data
transfer and optional KMS charges can still apply.

The commands intentionally fail until the two Terraform stacks are implemented
in Phase 0.2. Do not run `start` or `stop` without reviewing the Terraform plan.