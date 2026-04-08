---
title: "scikit-learn Pickle Version Mismatch Crashes FastAPI on Startup"
date: "2026-04-05"
category: "runtime-errors"
module: "ML Model Deployment"
problem_type: runtime_error
component: tooling
symptoms:
  - "AttributeError: Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer'>"
  - "InconsistentVersionWarning: Trying to unpickle estimator from version 1.6.1 when using version 1.7.2"
  - "FastAPI application startup failed. Exiting."
  - "Supervisor shows RUNNING but curl to /health returns empty response"
  - "Terraform provisioner health check times out after all attempts"
root_cause: incomplete_setup
resolution_type: dependency_update
severity: critical
related_components:
  - development_workflow
tags:
  - scikit-learn
  - pickle
  - version-mismatch
  - model-loading
  - ml-dependencies
  - deployment
  - supervisor
---

# scikit-learn Pickle Version Mismatch Crashes FastAPI on Startup

## Problem

The FastAPI ML application failed to start on an EC2 instance after deployment. The app entered a
crash-restart loop managed by Supervisor because `model.pkl` was saved with scikit-learn 1.6.1 but
`requirements.txt` pinned `scikit-learn==1.7.2`. The version mismatch caused an `AttributeError`
during model deserialization, blocking the entire application from starting.

## Symptoms

- `supervisorctl status predict-api` shows `RUNNING` (misleading — Supervisor auto-restarts the crashing process)
- `curl -s http://localhost:8000/health` returns empty response (app crashes before binding to port)
- `stderr.log` shows:
  ```
  InconsistentVersionWarning: Trying to unpickle estimator RandomForestRegressor from version 1.6.1
  when using version 1.7.2. This might lead to breaking code or invalid results.

  AttributeError: Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer'
  from '.../sklearn/compose/_column_transformer.py'>

  ERROR:    Application startup failed. Exiting.
  ```
- Terraform health check exhausts all retries and reports provisioner error

## What Didn't Work

- **Checking `supervisorctl status`** — showed `RUNNING`, which was misleading. Supervisor's
  `autorestart=true` was hiding the real issue by immediately restarting the crashing process.
  Always check the logs, not just the status.

- **Looking at the wrong log path** — checked `/var/log/predict-api/predict-api.log` which
  doesn't exist. The actual paths from `supervisor.conf` are:
  - `/var/log/predict-api/stdout.log`
  - `/var/log/predict-api/stderr.log`

## Solution

The fix is to pin scikit-learn in `requirements.txt` to the exact version used to train and
serialize the model.

**Immediate fix on the running server:**

```bash
/home/ubuntu/Project1/venv/bin/pip install scikit-learn==1.6.1
sudo supervisorctl restart predict-api
sleep 5
curl -s http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

**Permanent fix in version control (`requirements.txt`):**

```diff
- scikit-learn==1.7.2
+ scikit-learn==1.6.1
```

Commit and push so future deployments install the correct version.

## Why This Works

Python's pickle protocol serializes objects by storing their class references as full dotted module
paths (e.g., `sklearn.compose._column_transformer._RemainderColsList`). When a class is renamed,
moved, or removed between library versions, pickle cannot resolve the stored path and raises
`AttributeError: Can't get attribute '...'`.

Between scikit-learn 1.6.x and 1.7.x, the internal class `_RemainderColsList` in
`sklearn.compose._column_transformer` was restructured. A model saved (pickled) with 1.6.1 stores
a reference to this class. Loading it with 1.7.2 — where the class no longer exists at that path —
raises the `AttributeError` during deserialization, before the model object is ever returned.

Pinning scikit-learn to 1.6.1 restores the correct class paths, deserialization succeeds, and the
application starts normally.

## Prevention

- **Pin scikit-learn to the exact training version.** The version used to save `model.pkl` must
  match the version installed in the deployment environment. This applies to any library that
  scikit-learn stores references to internally (numpy, scipy, etc. are less risky but the same
  principle applies).

- **Update `requirements.txt` whenever you retrain.** If you retrain the model with a newer
  scikit-learn version, update the pin in `requirements.txt` in the same commit.

- **Store model metadata alongside the artifact.** Create a `ml/model_metadata.json` so the
  training environment is self-documenting:

  ```json
  {
    "trained_date": "2026-04-05",
    "scikit_learn_version": "1.6.1",
    "python_version": "3.10"
  }
  ```

  Load and log this at startup so version mismatches surface immediately in logs rather than as
  cryptic `AttributeError`s.

- **Check Supervisor logs, not just status.** `supervisorctl status` showing `RUNNING` does not
  mean the app is healthy — `autorestart=true` will keep restarting a crash-looping process
  indefinitely. Always check `stderr.log` when the health endpoint is unreachable.

## Related Issues

- `docs/plans/2026-04-04-001-feat-ec2-terraform-deployment-plan.md` — the deployment plan
  originally documented `scikit-learn==1.7.2`; this was the incorrect version that caused this bug.
  The plan is now superseded by the `scikit-learn==1.6.1` pin in `requirements.txt`.
