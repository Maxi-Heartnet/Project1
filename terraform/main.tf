terraform {
  required_version = ">= 1.4"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# AMI — latest Ubuntu 22.04 LTS (Jammy) x86_64 HVM SSD
# ---------------------------------------------------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------------------------------------------------------
# SSH key pair
# ---------------------------------------------------------------------------
resource "tls_private_key" "main" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "main" {
  key_name   = "project1-key"
  public_key = tls_private_key.main.public_key_openssh
}

resource "local_sensitive_file" "private_key" {
  content         = tls_private_key.main.private_key_pem
  filename        = "${path.module}/key.pem"
  file_permission = "0600"
}

# ---------------------------------------------------------------------------
# Security group — SSH + HTTP in, all out
# ---------------------------------------------------------------------------
resource "aws_security_group" "main" {
  name        = "project1-sg"
  description = "Allow SSH and HTTP inbound; all outbound."

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# EC2 instance — three provisioner phases (see plan Unit 2 for rationale)
# ---------------------------------------------------------------------------
resource "aws_instance" "main" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.main.key_name
  vpc_security_group_ids = [aws_security_group.main.id]

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = tls_private_key.main.private_key_pem
    host        = self.public_ip
  }

  # -------------------------------------------------------------------------
  # remote-exec #1 — packages, repo clone, venv, pip, mkdir
  # -------------------------------------------------------------------------
  provisioner "remote-exec" {
    inline = [
      # Wait for cloud-init to finish (cap at 120 s in case it stalls)
      "timeout 120 sudo cloud-init status --wait || true",

      # Stop unattended-upgrades and wait for any held dpkg lock to release
      "sudo systemctl stop unattended-upgrades || true",
      "while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do echo 'Waiting for dpkg lock...'; sleep 2; done",

      # System packages — curl is required by the health check in remote-exec #2
      "sudo apt-get update -q",
      "sudo apt-get install -y curl python3 python3-pip python3-venv git supervisor nginx",

      # Ensure supervisord is running (apt install enables it, but be explicit)
      "sudo systemctl enable supervisor",
      "sudo systemctl start supervisor || true",

      # Clone repository
      "git clone ${var.repo_url} /home/ubuntu/Project1",

      # Virtualenv + dependencies (no --quiet so failures are visible in logs)
      "python3 -m venv /home/ubuntu/Project1/venv",
      "/home/ubuntu/Project1/venv/bin/pip install -r /home/ubuntu/Project1/requirements.txt",

      # Ensure upload directories exist (data/ is gitignored, not created by clone)
      "mkdir -p /home/ubuntu/Project1/data /home/ubuntu/Project1/ml",
    ]
  }

  # -------------------------------------------------------------------------
  # file provisioners — upload gitignored data and model
  # (run after remote-exec #1 so target dirs exist)
  # -------------------------------------------------------------------------
  provisioner "file" {
    source      = "${path.module}/../data/listings.csv"
    destination = "/home/ubuntu/Project1/data/listings.csv"
  }

  provisioner "file" {
    source      = "${path.module}/../ml/model.pkl"
    destination = "/home/ubuntu/Project1/ml/model.pkl"
  }

  # -------------------------------------------------------------------------
  # remote-exec #2 — Supervisor + Nginx config + health check
  # (run after file provisioners so model and data are in place)
  # -------------------------------------------------------------------------
  provisioner "remote-exec" {
    inline = [
      # --- Supervisor ---
      "sudo cp /home/ubuntu/Project1/deploy/supervisor.conf /etc/supervisor/conf.d/predict-api.conf",
      "sudo mkdir -p /var/log/predict-api",
      "sudo supervisorctl reread",
      "sudo supervisorctl update",
      # Explicit start in case autostart=true hasn't triggered yet
      "sudo supervisorctl start predict-api || true",

      # --- Nginx config ---
      # printf uses single-quoted format string so $host / $remote_addr are
      # treated as literals by the shell and end up verbatim in the Nginx config.
      "printf 'server {\\n    listen 80;\\n    location / {\\n        proxy_pass http://127.0.0.1:8000;\\n        proxy_set_header Host $host;\\n        proxy_set_header X-Real-IP $remote_addr;\\n    }\\n}\\n' | sudo tee /etc/nginx/sites-available/predict-api > /dev/null",
      "sudo unlink /etc/nginx/sites-enabled/default || true",
      "sudo ln -s /etc/nginx/sites-available/predict-api /etc/nginx/sites-enabled/predict-api",
      "sudo nginx -t",
      "sudo systemctl reload nginx",

      # --- Health check (10 attempts × 3 s = 30 s total) ---
      "for i in $(seq 1 10); do echo \"Health check attempt $i/10...\"; curl -sf http://localhost:8000/health && exit 0; sleep 3; done; echo 'Health check failed after 10 attempts'; exit 1",
    ]
  }

  tags = {
    Name = "project1"
  }
}

# ---------------------------------------------------------------------------
# Elastic IP — stable public address across stop/start cycles
# ---------------------------------------------------------------------------
resource "aws_eip" "main" {
  instance = aws_instance.main.id
  domain   = "vpc"
}
