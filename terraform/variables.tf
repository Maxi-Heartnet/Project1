variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type. t3.small (2 vCPU, 2 GB) is the minimum comfortable size for this workload."
  type        = string
  default     = "t3.small"
}

variable "repo_url" {
  description = "HTTPS URL of the GitHub repository to clone onto the instance."
  type        = string
  default     = "https://github.com/Maxi-Heartnet/Project1.git"
}

variable "ssh_cidr" {
  description = <<-EOT
    CIDR block allowed to SSH into the instance (port 22).
    Defaults to 0.0.0.0/0 (open to the world) for convenience.
    Restrict to your own IP for any long-lived environment:
      ssh_cidr = "$(curl -s ifconfig.me)/32"
  EOT
  type        = string
  default     = "0.0.0.0/0"
}
