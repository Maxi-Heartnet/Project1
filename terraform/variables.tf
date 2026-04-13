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

variable "google_maps_api_key" {
  description = <<-EOT
    Google Maps JavaScript API key.
    Obtain from Google Cloud Console > APIs & Services > Credentials.
    Restrict to HTTP referrers (production domain) and Maps JavaScript API only.
    Add to terraform.tfvars (never commit that file).
  EOT
  type      = string
  sensitive = true
}

variable "maps_map_id" {
  description = <<-EOT
    Google Cloud Map ID required by AdvancedMarkerElement.
    Obtain from Google Cloud Console > Google Maps Platform > Map Management.
    Use "DEMO_MAP_ID" for local development (markers render but styling is limited).
    Add to terraform.tfvars (never commit that file).
  EOT
  type      = string
  sensitive = true
}
