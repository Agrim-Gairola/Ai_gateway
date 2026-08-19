terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region"    { default = "us-east-1" }
variable "instance_type" { default = "t3.medium" }
variable "key_name"      { description = "Your EC2 key pair name" }

# ── Security Group ────────────────────────────────────────────────────────────
resource "aws_security_group" "gateway_sg" {
  name        = "ai-gateway-sg"
  description = "AI Gateway security group"

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "FastAPI gateway"
  }
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Streamlit dashboard"
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── EC2 Instance ──────────────────────────────────────────────────────────────
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "gateway" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.gateway_sg.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y docker.io docker-compose-plugin git curl
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ubuntu

    git clone https://github.com/YOUR_USERNAME/ai-gateway.git /opt/ai-gateway
    cd /opt/ai-gateway

    # Set env vars from SSM or hardcode for demo
    export OPENAI_API_KEY="${var.openai_api_key}"
    export ANTHROPIC_API_KEY="${var.anthropic_api_key}"

    docker compose up -d --build
    echo "✅ AI Gateway deployed!"
  EOF

  tags = {
    Name        = "ai-gateway"
    Environment = "production"
    Project     = "enterprise-ai-gateway"
  }
}

variable "openai_api_key"    { sensitive = true; default = "mock" }
variable "anthropic_api_key" { sensitive = true; default = "mock" }

output "gateway_url"   { value = "http://${aws_instance.gateway.public_ip}:8000" }
output "dashboard_url" { value = "http://${aws_instance.gateway.public_ip}:8501" }
output "instance_id"   { value = aws_instance.gateway.id }
output "public_ip"     { value = aws_instance.gateway.public_ip }
