# Credentials Setup Guide

This guide explains how to configure API credentials for the Mistral Hackathon NYC workspace.

## Overview

Three credentials are required to run the components:
- **MISTRAL_API_KEY** — Mistral AI API authentication
- **NVIDIA_BEARER_TOKEN** — NVIDIA integrated API authentication  
- **DATABRICKS_PAT** — Databricks personal access token

Additional credentials are optional and enable specific features (W&B logging, ElevenLabs voice, HuggingFace model access).

## Setup Methods

### Method 1: GitHub Codespaces Secrets (Recommended)

Secrets added to GitHub Codespaces are automatically available as environment variables in your dev container.

**Steps:**

1. Go to your GitHub repository settings → Secrets and variables → Codespaces
2. Create three repository secrets:
   - Name: `MISTRAL_API_KEY` | Value: Your Mistral API key
   - Name: `NVIDIA_BEARER_TOKEN` | Value: Your NVIDIA bearer token
   - Name: `DATABRICKS_PAT` | Value: Your Databricks PAT

3. Close and reopen your Codespace (or restart the dev container)

4. Verify secrets are loaded:
   ```bash
   echo $MISTRAL_API_KEY  # Should show first 10 chars of your key
   echo $NVIDIA_BEARER_TOKEN
   echo $DATABRICKS_PAT
   ```

**Advantages:**
- ✅ Automatic injection into all terminals and apps
- ✅ No credentials in version control
- ✅ Shared across all Codespaces in the repo
- ✅ GitHub handles secret masking in logs

### Method 2: Local `.env` File (Development)

For local development outside Codespaces, use a `.env` file.

**Steps:**

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```dotenv
   MISTRAL_API_KEY=sk-...
   NVIDIA_BEARER_TOKEN=...
   DATABRICKS_PAT=da...
   ```

3. The framework automatically loads `.env` when you run Python scripts or Streamlit apps

**⚠️ Security Warning:**
- Never commit `.env` to version control
- `.env` is already in `.gitignore` (safe)
- For team development, use Codespaces secrets instead

### Method 3: Environment Variables (Manual)

For CI/CD or custom deployments, set environment variables directly:

```bash
export MISTRAL_API_KEY="your-key-here"
export NVIDIA_BEARER_TOKEN="your-token-here"
export DATABRICKS_PAT="your-pat-here"
```

## Credential Loading Order

The framework follows this priority when loading credentials:

1. **Environment variables** (set via Codespaces secrets or manual `export`)
2. **`.env` file** (loaded automatically from root directory)
3. **Missing** → Error with helpful message

This means Codespaces secrets override `.env` values, allowing flexibility for different environments.

## Getting Your Credentials

### MISTRAL_API_KEY

1. Go to [console.mistral.ai](https://console.mistral.ai)
2. Sign in or create an account
3. Navigate to **API Keys** section
4. Click **Create API Key**
5. Copy the key and add to Codespaces secrets or `.env`

**URL:** https://console.mistral.ai/api-keys/

### NVIDIA_BEARER_TOKEN

1. Go to [integrate.api.nvidia.com](https://integrate.api.nvidia.com)
2. Sign in or create an account (NVIDIA developer)
3. Navigate to **API Keys** or **Authentication** section
4. Generate/copy your bearer token
5. Add to Codespaces secrets or `.env`

**URL:** https://integrate.api.nvidia.com

### DATABRICKS_PAT

1. Log into your Databricks workspace
2. Click on your **user profile** icon (top right)
3. Select **Settings**
4. Go to **Developer** → **Personal access tokens**
5. Click **Generate new token**
6. Copy the token (shown once)
7. Add to Codespaces secrets or `.env`

**Note:** Replace `<workspace-url>` with your Databricks URL

## Using Credentials in Code

### Option 1: Using Credential Module (Recommended)

```python
from credentials import load_credentials, validate_credentials

# Load credentials from .env or environment
creds = load_credentials()

# Validate required credentials exist
validate_credentials(creds, required=["MISTRAL_API_KEY", "NVIDIA_BEARER_TOKEN"])

# Access individual credentials
mistral_key = creds.get("MISTRAL_API_KEY")
```

### Option 2: Direct Environment Access (Legacy)

```python
import os

mistral_key = os.getenv("MISTRAL_API_KEY")
if not mistral_key:
    raise ValueError("MISTRAL_API_KEY not set")
```

### Option 3: In Streamlit Apps

Streamlit apps automatically load `.env` and environment variables:

```python
import streamlit as st
from credentials import load_credentials

creds = load_credentials()
mistral_key = creds.get("MISTRAL_API_KEY")

st.write(f"API Key loaded: {bool(mistral_key)}")
```

## Testing Credential Setup

Run the DataBolt-Edge test UI to verify all credentials are loaded:

```bash
# Install UI dependencies
uv sync --extra ui

# Run Streamlit test app
uv run streamlit run UI/streamlit_test_ui.py
```

The app will show:
- ✅ **Success** if `NVIDIA_BEARER_TOKEN` is set
- ❌ **Failed** with error message if credentials are missing

## Troubleshooting

### Error: "MISTRAL_API_KEY is not set"

**Solution:**
1. In Codespaces: Check that the secret is set (Settings → Secrets)
2. Locally: Verify `.env` file exists and contains the key
3. Test with: `echo $MISTRAL_API_KEY`

### Error: "Missing NVIDIA API token"

**Solution:**
1. Ensure `NVIDIA_BEARER_TOKEN` (not `NVIDIA_API_KEY`) is set
2. The variable name was changed to match NVIDIA's API conventions
3. Update your Codespaces secret name if using the old name

### Credentials Work Locally but Not in Codespaces

**Solution:**
1. Restart the Codespace (close and reopen)
2. Verify the secret was created in repo settings (not personal)
3. Check that secret is available to Codespaces (not just selected repos)

### Credentials Work in One App But Not Another

**Solution:**
1. Apps that use `load_dotenv()` automatically load `.env`
2. Apps that don't call `load_dotenv()` rely on environment variables alone
3. Check if the app is in a subdirectory that can't find `.env` in root
4. Use `credentials.load_credentials(env_file=Path(__file__).parent.parent / ".env")`

## Security Best Practices

1. **Never commit credentials** — Use `.gitignore` (it's already configured)
2. **Use Codespaces secrets for team work** — Safer than `.env`
3. **Rotate keys regularly** — Especially after sharing/demos
4. **Use minimal-permission tokens** — Request only necessary API scopes
5. **Mask secrets in logs** — GitHub Actions automatically masks Codespaces secrets
6. **Audit API usage** — Check provider dashboards for unexpected activity

## For CI/CD (GitHub Actions)

Secrets used in GitHub Actions workflows are automatically available:

```yaml
- name: Run tests
  env:
    MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
    NVIDIA_BEARER_TOKEN: ${{ secrets.NVIDIA_BEARER_TOKEN }}
    DATABRICKS_PAT: ${{ secrets.DATABRICKS_PAT }}
  run: uv run pytest
```

No need to modify code — environment variables are injected automatically.

## Additional Resources

- [Mistral AI Docs](https://docs.mistral.ai/)
- [NVIDIA API Integration](https://integrate.api.nvidia.com)
- [Databricks API Tokens](https://docs.databricks.com/en/dev-tools/auth)
- [GitHub Codespaces Secrets](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-secrets-for-your-codespaces)
- [Python dotenv Documentation](https://python-dotenv.readthedocs.io/)
