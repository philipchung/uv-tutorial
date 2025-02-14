# UV Tutorial

Boilerplate code for a project using huggingface models that uses uv monorepo.

This uses uv workspaces to create a project with multiple subpackages that can be imported into your main project scripts. 

Declaration of dependencies is per-package, but all dependencies are solved together to create a lock file which is then used to build the python environment.

Overall, this repo tutorial shows you:
1. How to use `uv` to create a monorepo with a main project/package code and several helper packages that can be imported into the main project
2. How to create a virtual environment managed by `uv`
3. How to solve dependencies across all of your packages in the virtual environment
4. How to sync solved dependencies into your virtual environment
5. How to specify a specific CUDA version of pytorch using `uv`
6. How to use an `.env` environment file and automatically import this into `os.environ` dictionary in python
7. A bunch of random utility functions
8. How to constrain visibility of GPUs to specific devices on a compute node
9. How to load a Distilled Deepseek R1 model and Run Inference using transformers library
10. How to Generate a push notification to your slack channel when job completes

```sh
# Add package to main project dependencies (to add to specific package, cd into package dir)
uv add package_xyz
# Make virtual environment
uv venv
# Solve all dependencies to make a cross-platform lockfile
uv lock
# Sync lock file for all packages into your python environment
uv sync --all-packages
```


```sh
## Example Notebooks

# Shows you how to see which GPUs are available to your script.
src/notebooks/gpus_unconstrained.py
# Shows you how to constrain your script's visibility of available GPU devices
src/notebooks/gpus_constrained.py

# Sample Script to Simulate a Job
# NOTE: Replace environment variable `SLACK_WEBHOOK_URL`
# 1. Constrain GPUs to GPU1
# 2. Load Distilled Deepseek R1 model on GPU
# 3. Run Inference
# 4. Generate a Notification to your Slack Channel when the Job Completes
src/notebooks/run_r1.py
```
