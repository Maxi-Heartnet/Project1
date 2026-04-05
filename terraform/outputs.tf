output "elastic_ip" {
  description = "Elastic IP address of the EC2 instance. Open http://<elastic_ip>/ in a browser."
  value       = aws_eip.main.public_ip
}
