---
date: 2026-04-04
topic: ec2-terraform-deploy
---

# EC2 Terraform Deployment

## Problem Frame

The Santo Domingo House Price API and web frontend run manually on an EC2 instance with no
reproducible provisioning. Every new instance requires hand-running apt, git clone, pip install,
and Supervisor/Nginx config steps from memory. The goal is a single `terraform apply` that
provisions and fully configures a fresh EC2 instance so the app is reachable at port 80 with
no manual steps.

## Architecture

```
  Developer machine
  ┌─────────────────────────────────────────────────────────┐
  │  terraform apply                                        │
  │  ├── creates EC2, SG, EIP, key pair                    │
  │  ├── remote-exec #1 → installs packages, clones repo,  │
  │  │                     creates venv, pip install,       │
  │  │                     mkdir -p data/ ml/               │
  │  ├── file provisioner → uploads listings.csv,           │
  │  │                       model.pkl                      │
  │  └── remote-exec #2 → configures Nginx + Supervisor,   │
  │                        starts services, health-check    │
  └─────────────────────────────────────────────────────────┘
                          │
                     EC2 (t3.small)
                          │
              ┌───────────┴───────────┐
              │         Nginx :80     │
              │           │           │
              │    Uvicorn :8000      │
              │   (2 workers, via     │
              │    Supervisor)        │
              │           │           │
              │     FastAPI app       │
              └───────────────────────┘
                          │
                    Elastic IP
                    (stable address)
```

## Requirements

**Infrastructure**

- R1. Terraform creates an EC2 `t3.small` instance running Ubuntu 22.04 LTS, using a data source
  to resolve the latest AMI by name so it stays current without manual updates.
  (AMI filter: `ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*`, owner `099720109477`.)
- R2. Terraform creates and attaches an Elastic IP so the public address is stable across
  stop/start cycles.
- R3. Terraform generates a new RSA key pair, registers it with AWS, and saves the private key
  to `terraform/key.pem` (chmod 600). `terraform/key.pem` is added to `.gitignore`.
- R4. A security group allows:
  - Port 22 (SSH) inbound — defaults to `0.0.0.0/0`; lockable to a specific CIDR via variable
  - Port 80 (HTTP) inbound from `0.0.0.0/0`
  - All outbound traffic
- R5. Terraform outputs the Elastic IP address after apply so the URL is immediately visible.
- R6. All configurable values — AWS region, instance type, GitHub repo URL, SSH CIDR — are
  Terraform variables with sensible defaults. A `terraform.tfvars.example` documents required
  overrides.

**Provisioning — remote-exec block #1 (runs before file provisioners)**

- R7. Installs system packages: `python3`, `python3-pip`, `python3-venv`, `git`, `supervisor`,
  `nginx`.
- R8. Clones `https://github.com/Maxi-Heartnet/Project1.git` to `/home/ubuntu/Project1`.
- R9. Creates a virtualenv at `/home/ubuntu/Project1/venv` and runs
  `pip install -r requirements.txt`.
  - **Note**: `requirements.txt` currently pins `scikit-learn==1.7.2`. Before implementation,
    verify this version exists on PyPI (`pip index versions scikit-learn`). If it does not,
    the version pin must be corrected before provisioning will succeed.
- R10. Runs `mkdir -p /home/ubuntu/Project1/data /home/ubuntu/Project1/ml` to ensure upload
  target directories exist. (`data/` is not created by `git clone` because `data/listings.csv`
  is gitignored. `ml/` is created by the clone, but the mkdir is included for safety.)

**Data and model delivery (file provisioners — run after remote-exec #1)**

- R11. Uploads `data/listings.csv` from the local machine to
  `/home/ubuntu/Project1/data/listings.csv` on the server.
- R12. Uploads `ml/model.pkl` from the local machine to `/home/ubuntu/Project1/ml/model.pkl`
  on the server.

**Provisioning — remote-exec block #2 (runs after file provisioners)**

- R13. Copies `deploy/supervisor.conf` from the cloned repo to
  `/etc/supervisor/conf.d/predict-api.conf` using a shell `cp` command (no separate file
  provisioner — the file is already on the server from the `git clone` in R8). Creates
  `/var/log/predict-api/`, reloads Supervisor, and starts the `predict-api` service.
  - The Supervisor config hardcodes `/home/ubuntu/Project1/venv/bin/uvicorn` as the command
    path, which matches the virtualenv created in R9. ✓ (confirmed via `deploy/supervisor.conf`.)
- R14. Writes an Nginx site config to `/etc/nginx/sites-available/predict-api` that proxies
  port 80 → `127.0.0.1:8000`, passing `Host` and `X-Real-IP` headers. Removes the default
  site (`unlink /etc/nginx/sites-enabled/default`), enables the new site
  (`ln -s /etc/nginx/sites-available/predict-api /etc/nginx/sites-enabled/`), and reloads
  Nginx.
- R15. Final step: polls `curl -sf http://localhost:8000/health` with a short retry loop
  (e.g., 5 attempts × 3-second sleep) to confirm the app started successfully before
  `terraform apply` declares success. If the health check fails, remote-exec exits non-zero
  and Terraform reports a provisioning error.

**Repository layout**

- R16. All Terraform files live in a `terraform/` directory at the repo root.
  Structure: `main.tf`, `variables.tf`, `outputs.tf`, `terraform.tfvars.example`.
  `terraform/key.pem` and `terraform/.terraform/` are added to `.gitignore`.

## Success Criteria

- `terraform apply` runs to completion with no errors and no manual SSH steps required.
- `curl http://<EIP>/health` returns `{"status":"ok","model_loaded":true}`.
- The prediction form loads in a browser at `http://<EIP>/`.
- `terraform destroy` cleanly removes all created resources (EIP, EC2, SG, key pair).

## Scope Boundaries

- No HTTPS / TLS in this iteration (Nginx listens on port 80 only). Adding Certbot later
  is straightforward given the Nginx foundation.
- No auto-scaling, load balancer, or multi-instance setup.
- No RDS or external storage — model and data stay on the instance.
- No CI/CD pipeline — deployment is triggered manually via `terraform apply`.
- No remote state backend — `terraform.tfstate` lives locally. If `terraform.tfstate` is lost,
  resources must be cleaned up manually via the AWS Console.
- SSH key pair is generated fresh; no support for importing an existing key pair.

## Key Decisions

- **t3.small over t3.micro**: 2 GB RAM gives comfortable headroom for model load + 2 Uvicorn
  workers without OOM risk.
- **file provisioner over S3 or adding CSV to repo**: Avoids new infrastructure and keeps
  the repo's data-exclusion policy unchanged. The trade-off is that `terraform apply` must
  run from a machine that has both `data/listings.csv` and `ml/model.pkl` locally.
- **Nginx reverse proxy over direct port 8000**: Port 80 access, cleaner URLs, and a clear
  path to HTTPS via Certbot later — minimal extra complexity.
- **remote-exec over user_data**: remote-exec runs after the instance is reachable over SSH
  and reports errors to the Terraform run. `user_data` runs asynchronously with no error
  visibility, making failures hard to debug.
- **Two remote-exec blocks with file provisioners in between**: Terraform executes provisioners
  on a resource in declaration order. Placing file provisioners between the two remote-exec
  blocks guarantees that directories exist before uploads and that services are started after
  files are in place.
- **Local tfstate**: Solo project; remote state adds complexity with no benefit here.

## Dependencies / Assumptions

- AWS credentials are configured locally (`~/.aws/credentials` or environment variables).
- The machine running `terraform apply` has `data/listings.csv` and `ml/model.pkl` present.
- The GitHub repo `https://github.com/Maxi-Heartnet/Project1.git` is public (no deploy key
  needed for `git clone` over HTTPS).
- Terraform CLI is installed locally.
- `scikit-learn==1.7.2` is available on PyPI. (Verify before implementation — see R9.)

## Outstanding Questions

### Deferred to Planning

- [Affects R9][Needs research] Verify `scikit-learn==1.7.2` exists on PyPI before writing
  provisioning scripts. If it does not, determine the correct pinned version.
- [Affects R4][User decision] Consider defaulting the SSH CIDR variable to the developer's
  current public IP rather than `0.0.0.0/0` to reduce attack surface.

## Next Steps

→ `/ce:plan` for structured implementation planning
