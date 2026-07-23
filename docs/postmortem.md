# Postmortem — Streamlit Cloud FileNotFoundError

## What happened

After building and testing the LoanLens Streamlit app locally — where it 
worked correctly — I deployed it to Streamlit Community Cloud. The app 
immediately crashed with:
FileNotFoundError: [Errno 2] No such file or directory: 'lightgbm_model.pkl'
## Root cause

Locally, I ran `streamlit run app.py` from inside the `app/` directory, so 
a relative path like `joblib.load("lightgbm_model.pkl")` correctly resolved 
to a file sitting right next to the script. On Streamlit Cloud, the app is 
launched from the repository root (`/mount/src/loanlens`), not from inside 
`app/` — so the same relative path no longer pointed anywhere valid.

This is a classic "works on my machine" bug: my code was never actually 
wrong, it just carried an unstated assumption (the working directory) that 
happened to hold locally and silently broke once the deployment environment 
changed it.

## The fix

I anchored all file paths to the script's own location instead of relying 
on the working directory:

```python
import os
APP_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(APP_DIR, "lightgbm_model.pkl"))
```

This works identically regardless of where the script is launched from, 
since `__file__` always points to the script's own location on disk.

## What I'd do differently next time

I'd write file-loading code this way by default, from the start, rather 
than only fixing it after a deployment failure. More broadly, I'd treat 
"does this assume a specific working directory?" as a standard checklist 
item before any deployment, the same way I'd check for hardcoded 
credentials or missing environment variables.
