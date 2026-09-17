# Publish this project on GitHub

Suggested repository name:

`ml-cybersecurity-threat-detection-cicids2017`

## Option A: GitHub website

1. Create a new **public or private repository** with the suggested name.
2. Do not add a second README or license during repository creation, because this project already contains its own README.
3. Extract/copy this project folder.
4. Upload the project files.
5. After publishing, copy the repository URL into the weekly report.

## Option B: Git command line

From the project folder:

```bash
git init
git add .
git commit -m "Initial CICIDS2017 cybersecurity ML implementation"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/ml-cybersecurity-threat-detection-cicids2017.git
git push -u origin main
```

Replace `<YOUR-USERNAME>` with your GitHub username.

## What should be committed

Commit the notebook, source-code export, README, requirements, documentation, and actual results/figures/model artifacts **after the notebook has successfully run**.

Do **not** commit:

- `kaggle.json` or any API credential
- raw CICIDS2017 CSV files
- personal/private files

The `.gitignore` already excludes the credential and raw CSV data.
