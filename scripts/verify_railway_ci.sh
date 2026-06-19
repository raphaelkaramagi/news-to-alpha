#!/usr/bin/env bash
# Verify the same Railway auth path used by .github/workflows/daily-update.yml
#
# Usage (paste a fresh Account token from railway.com/account/tokens):
#   export RAILWAY_TOKEN="<account-token>"
#   export RAILWAY_PROJECT_ID="91bd0f77-9c80-416e-8f60-8409ae0f0927"
#   export RAILWAY_SERVICE="web"
#   export RAILWAY_ENVIRONMENT="production"
#   export RAILWAY_SSH_KEY_PATH="$HOME/.ssh/railway_github_actions"
#   bash scripts/verify_railway_ci.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK:   $*"; }

for var in RAILWAY_TOKEN RAILWAY_PROJECT_ID RAILWAY_SERVICE RAILWAY_ENVIRONMENT RAILWAY_SSH_KEY_PATH; do
  if [ -z "${!var:-}" ]; then
    fail "Set $var (see script header for example values)"
  fi
done

[ -f "$RAILWAY_SSH_KEY_PATH" ] || fail "SSH key not found: $RAILWAY_SSH_KEY_PATH"

if ! command -v railway >/dev/null; then
  fail "Install Railway CLI: npm install -g @railway/cli"
fi

echo "=== Railway CI verify ==="
echo "CLI: $(railway --version)"
echo "Token length: ${#RAILWAY_TOKEN} chars (expect ~36 for UUID-style token)"
echo "SSH key: $RAILWAY_SSH_KEY_PATH"
ssh-keygen -lf "${RAILWAY_SSH_KEY_PATH}.pub" 2>/dev/null || ssh-keygen -lf "$RAILWAY_SSH_KEY_PATH" 2>/dev/null || true
echo "Project: $RAILWAY_PROJECT_ID  Service: $RAILWAY_SERVICE  Env: $RAILWAY_ENVIRONMENT"
echo

# Must match GitHub Actions: token auth + registered SSH key
if ! railway whoami; then
  fail "railway whoami failed — token invalid/expired. Create Account token at https://railway.com/account/tokens (No workspace) and export RAILWAY_TOKEN"
fi
ok "railway whoami"

SSH=(railway ssh
  -p "$RAILWAY_PROJECT_ID"
  -s "$RAILWAY_SERVICE"
  -e "$RAILWAY_ENVIRONMENT"
  -i "$RAILWAY_SSH_KEY_PATH")

if ! "${SSH[@]}" -- echo "ssh ok"; then
  fail "railway ssh failed — check RAILWAY_SSH_PRIVATE_KEY in GitHub matches this private key"
fi
ok "railway ssh"

echo
echo "/data on Railway:"
"${SSH[@]}" -- ls -la /data

ITEMS=$("${SSH[@]}" -- sh -c 'cd /data && for p in database.db processed models; do [ -e "$p" ] && printf "%s " "$p"; done')
[ -n "$ITEMS" ] || fail "No pullable artifacts under /data"
ok "pull targets: $ITEMS"

echo
echo "All checks passed — GitHub Actions should work if secrets match these values."
