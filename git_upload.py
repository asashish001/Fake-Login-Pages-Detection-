import os
import subprocess
import sys

# Configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/asashish001/Fake-Login-Pages-Detection-"
COMMIT_MESSAGE = "Add full project files (dataset, models, code)"
GIT_EMAIL = "asashish001@users.noreply.github.com"
GIT_NAME = "asashish001"

def run_command(cmd, ignore_errors=False):
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        if not ignore_errors:
            print(f"Command failed: {cmd}")
            sys.exit(1)
        else:
            print(f"Command failed but proceeding (ignore_errors=True).")

def main():
    print(f"Processing repository at: {PROJECT_DIR}")

    # 0. Initialize Git if needed
    if not os.path.isdir(os.path.join(PROJECT_DIR, ".git")):
        print("No .git directory found. Initializing repository...")
        run_command("git init")
        run_command("git branch -M main")
        run_command(f"git remote add origin {REPO_URL}")

    # Configure git user for this repository to fix "Author identity unknown"
    run_command(f'git config user.email "{GIT_EMAIL}"')
    run_command(f'git config user.name "{GIT_NAME}"')

    # 1. Add all files (Dataset, models, code, etc.)
    print("Adding all files to staging...")
    run_command("git add .")

    # 3. Commit
    # ignore_errors=True because git commit fails if there is nothing new to commit
    run_command(f'git commit -m "{COMMIT_MESSAGE}"', ignore_errors=True)

    # 4. Push
    print("Attempting to push to remote...")
    # Try pushing to main; if it fails, you might need to set up the remote or credentials manually
    # or try 'master' if your branch is named differently.
    run_command("git push origin main --force", ignore_errors=True)
    print("Done.")

if __name__ == "__main__":
    main()