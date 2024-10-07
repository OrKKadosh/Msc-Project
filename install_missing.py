# Read the old environment packages
import os

with open('old_env_packages.txt', 'r') as f_old:
    old_packages = {line.split('=')[0].strip() for line in f_old if '=' in line}

# Read the new environment packages
with open('new_env_packages.txt', 'r') as f_new:
    new_packages = {line.split('=')[0].strip() for line in f_new if '=' in line}

# Identify missing packages
missing_packages = old_packages - new_packages

# Print and install missing packages
print(f"Missing packages: {missing_packages}")
for package in missing_packages:
    print(f"Installing {package}...")
    os.system(f"pip install {package}")
