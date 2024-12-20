#!/bin/bash

# Update system and install Git
echo "Updating system and installing Git..."
sudo apt update && sudo apt install git -y

# Prompt user for Personal Access Token (PAT)
read -p "Enter your GitHub Personal Access Token (PAT): " PAT

# Clone repository using PAT (replace with your repo URL)
echo "Cloning repository using Personal Access Token..."
read -p "Enter your repository URL: " REPO_URL
git clone https://$PAT@${REPO_URL#https://}

# Set Git global configuration
echo "Setting up Git global configuration..."
git config --global user.name "kafkapple"
git config --global user.email "biasdrive@gmail.com"
git config --global core.editor vim
git config --global core.pager cat
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

# Enable credential helper for storing credentials
echo "Enabling Git credential helper..."
git config --global credential.helper store

# Optional: Set upstream remote URL
read -p "Do you want to configure an upstream remote? (y/n): " CONFIGURE_UPSTREAM
if [[ "$CONFIGURE_UPSTREAM" == "y" ]]; then
  read -p "Enter your upstream repository URL (HTTPS or SSH): " UPSTREAM_URL
  git remote set-url upstream $UPSTREAM_URL
fi

# Check and display Git configuration
echo "Git global configuration:"
git config --list --show-origin

# Reset Git configuration (optional)
read -p "Do you want to reset your Git configuration? (y/n): " RESET_CONFIG
if [[ "$RESET_CONFIG" == "y" ]]; then
  if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    rm ~/.gitconfig
    echo "Git configuration reset (Linux/Mac)."
  elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win32" ]]; then
    del %USERPROFILE%\.gitconfig
    echo "Git configuration reset (Windows)."
  fi
fi

echo "Git setup complete!"

#chmod +x git_setup.sh
#bash git_setup.sh

