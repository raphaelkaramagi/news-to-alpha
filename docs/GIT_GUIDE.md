# Git & GitHub Team Guide

A step-by-step reference for every team member.  

---

## Table of Contents

1. [Why branches & PRs?](#1-why-branches--prs)
2. [One-time setup](#2-one-time-setup)
3. [Daily workflow (the loop)](#3-daily-workflow-the-loop)
4. [Step-by-step: IDE (VS Code / Cursor)](#4-step-by-step-ide)
5. [Step-by-step: Terminal](#5-step-by-step-terminal)
6. [Step-by-step: GitHub website](#6-step-by-step-github-website)
7. [Keeping your branch up-to-date](#7-keeping-your-branch-up-to-date)
8. [Handling merge conflicts](#8-handling-merge-conflicts)
9. [Commit message conventions](#9-commit-message-conventions)
10. [Cheat sheet](#10-cheat-sheet)

---

## 1. Why branches & PRs?

| Without branches | With branches |
|-----------------|---------------|
| Everyone pushes to `main` | Each person works on their own copy |
| One bad push breaks everyone | `main` always works |
| No code review | Teammates review before merging |
| Hard to track who did what | Clear history per feature |

**The rule**: Nobody pushes directly to `main`.  
All changes go through a **feature branch** → **Pull Request** → **review** → **merge**.

---

## 2. One-time setup

Make sure you've already cloned the repo:

```bash
git clone https://github.com/raphaelkaramagi/news-to-alpha.git
cd news-to-alpha
```

Verify your remote is set:

```bash
git remote -v
# Should show:
#   origin  https://github.com/raphaelkaramagi/news-to-alpha.git (fetch)
#   origin  https://github.com/raphaelkaramagi/news-to-alpha.git (push)
```

> **Tip**: If you haven't configured your Git identity yet:
> ```bash
> git config --global user.name "Your Name"
> git config --global user.email "your-email@duke.edu"
> ```

---

## 3. Daily workflow (the loop)

```
1. Pull latest main
2. Create a feature branch
3. Write code, commit often
4. Push your branch
5. Open a Pull Request on GitHub
6. Get a review from a teammate
7. Merge the PR
8. Delete the branch
9. Pull latest main again
```

Below are the detailed steps for each part.

---

## 4. Step-by-step: IDE (VS Code / Cursor)

### 4a. Pull latest main

1. Click the branch name in the **bottom-left** corner of the status bar.
2. Select `main` from the dropdown.
3. Open the **Source Control** panel (left sidebar, branch icon) or press `Ctrl+Shift+G` / `Cmd+Shift+G`.
4. Click the `...` menu → **Pull**.

### 4b. Create a feature branch

1. Click the branch name in the bottom-left.
2. Select **"Create new branch…"**.
3. Type your branch name using this pattern:

   ```
   feature/short-description
   ```

   Examples:
   - `feature/price-collector`
   - `feature/news-validation`
   - `fix/rate-limit-bug`
   - `docs/update-readme`

4. Press Enter. You're now on your new branch.

### 4c. Write code & commit

1. Make your changes to files.
2. Open Source Control panel.
3. You'll see changed files listed.
4. Click the **+** button next to each file to **stage** it (or click + on "Changes" header to stage all).
5. Type a commit message in the text box at the top (see [conventions below](#9-commit-message-conventions)).
6. Click the **checkmark** button (or `Cmd+Enter`) to commit.
7. Repeat as needed — commit often, at least once per logical change.

### 4d. Push your branch

1. After committing, click the `...` menu in Source Control → **Push**.
2. If it asks about setting upstream, say **Yes** / **OK**.
3. Your branch is now on GitHub.

### 4e. Open a Pull Request

1. Go to GitHub in your browser (see [Section 6](#6-step-by-step-github-website)).
2. Or: the IDE may show a notification with "Create Pull Request" — click it.

---

## 5. Step-by-step: Terminal

### 5a. Pull latest main

```bash
git checkout main
git pull origin main
```

### 5b. Create a feature branch

```bash
git checkout -b feature/your-feature-name
```

This creates the branch AND switches to it in one command.

### 5c. Write code & commit

```bash
# Check what changed
git status

# Stage specific files
git add src/data_collection/price_collector.py
git add tests/unit/test_price_collector.py

# Or stage everything
git add .

# Commit with a message
git commit -m "feat: add price collector with yfinance integration"
```

### 5d. Push your branch

```bash
# First push (sets up tracking)
git push -u origin feature/your-feature-name

# Subsequent pushes (just)
git push
```

### 5e. Open a Pull Request

```bash
# If you have the GitHub CLI (gh) installed:
gh pr create --title "Add price data collector" --body "Implements PriceCollector class with retry logic and deduplication."

# Otherwise, go to GitHub.com (see next section)
```

---

## 6. Step-by-step: GitHub website

### 6a. Open a Pull Request

1. Go to **https://github.com/raphaelkaramagi/news-to-alpha**.
2. You'll see a yellow banner: **"feature/your-branch had recent pushes — Compare & pull request"**. Click it.
   - If you don't see the banner, click the **"Pull requests"** tab → **"New pull request"**.
   - Set **base**: `main` and **compare**: `feature/your-branch`.
3. Fill in:
   - **Title**: Short description (e.g., "Add price data collector")
   - **Description**: What you did, how to test it, any concerns
   - **Reviewers**: Assign your partner (right sidebar)
     - Raphael ↔ Tim (financial team)
     - Moses ↔ Gordon (news team)
     - Cheri reviews integration / cross-team PRs
4. Click **"Create pull request"**.

### 6b. Review a teammate's PR

1. Go to **Pull requests** tab.
2. Click the PR you need to review.
3. Go to the **"Files changed"** tab to see the code diff.
4. Click on any line to leave a comment.
5. When done, click **"Review changes"** (top right of files tab):
   - **Approve**: Looks good, merge it.
   - **Request changes**: Something needs fixing first.
   - **Comment**: General feedback, no blocking.
6. Submit your review.

### 6c. Merge the PR

1. Once approved, go back to the **"Conversation"** tab.
2. Click the green **"Merge pull request"** button.
3. Click **"Confirm merge"**.
4. Click **"Delete branch"** (cleans up the remote branch).

### 6d. Update your local machine after merge

```bash
git checkout main
git pull origin main

# Delete your old local branch (optional, keeps things tidy)
git branch -d feature/your-feature-name
```

---

## 7. Keeping your branch up-to-date

If `main` has changed while you were working on your branch:

```bash
# Make sure your changes are committed first
git add .
git commit -m "wip: save progress before updating"

# Fetch and merge the latest main into your branch
git fetch origin
git merge origin/main
```

Alternatively, use rebase (cleaner history but slightly more complex):

```bash
git fetch origin
git rebase origin/main
```

> **When to do this**: Before opening a PR, or if GitHub shows
> "This branch has conflicts that must be resolved."

---

## 8. Handling merge conflicts

Conflicts happen when two people edit the same lines. Don't panic.

### In the terminal

```bash
# After a merge/rebase shows CONFLICT:
git status
# Shows which files have conflicts

# Open each conflicted file. You'll see markers like:
# <<<<<<< HEAD
# your version of the code
# =======
# their version of the code
# >>>>>>> origin/main

# Edit the file to keep the correct version (remove the markers).
# Then:
git add path/to/resolved-file.py
git commit -m "fix: resolve merge conflict in price_collector"
```

### In VS Code / Cursor

1. Conflicted files show up with a **"C"** badge in Source Control.
2. Open the file — you'll see colored blocks:
   - Green = your changes (Current Change)
   - Blue = their changes (Incoming Change)
3. Click one of the options above each block:
   - **Accept Current Change** — keep yours
   - **Accept Incoming Change** — keep theirs
   - **Accept Both Changes** — keep both
   - Or manually edit to combine them.
4. Save the file, stage it, and commit.

### Prevention tips

- Pull from main frequently (daily).
- Communicate with your partner about which files you're editing.
- Keep PRs small and focused.

---

## 9. Commit message conventions

We use **conventional commits** so the history is easy to read:

```
<type>: <short description>
```

| Type | When to use | Example |
|------|------------|---------|
| `feat` | New feature | `feat: add news collector with Finnhub API` |
| `fix` | Bug fix | `fix: handle empty yfinance response` |
| `docs` | Documentation | `docs: update README with setup instructions` |
| `refactor` | Code restructure (no new feature) | `refactor: extract retry logic to base class` |
| `test` | Adding / fixing tests | `test: add unit tests for schema creation` |
| `chore` | Maintenance, config | `chore: update .gitignore with model files` |

### Good commit messages

```
feat: add price data collector with retry logic
fix: prevent duplicate news inserts on re-run  
docs: add data collection instructions to README
test: add validation tests for cutoff-time logic
```

### Bad commit messages

```
update stuff
fixed it
asdfasdf
changes
WIP
```

---

## 10. Cheat sheet

### Starting a new feature

```bash
git checkout main
git pull origin main
git checkout -b feature/my-feature
# ... code ...
git add .
git commit -m "feat: description"
git push -u origin feature/my-feature
# → Open PR on GitHub
```

### Reviewing + merging (on GitHub)

```
Pull requests tab → select PR → Files changed → Review → Approve
→ Conversation tab → Merge pull request → Delete branch
```

### After a PR is merged

```bash
git checkout main
git pull origin main
git branch -d feature/my-feature
```

### Quick status check

```bash
git status          # What's changed?
git log --oneline   # Recent commits
git branch          # What branch am I on?
git branch -a       # All branches (local + remote)
```

---

## Branch naming by team member

| Person | Branch examples |
|--------|----------------|
| Raphael | `feature/price-collector`, `feature/technical-indicators`, `feature/lstm-model` |
| Tim | `feature/database-schema`, `feature/sequence-generator`, `feature/lstm-trainer` |
| Moses | `feature/news-collector`, `feature/nlp-baseline`, `feature/nlp-advanced` |
| Gordon | `feature/finnhub-api-client`, `feature/nlp-preprocessing`, `feature/embeddings` |
| Cheri | `feature/project-structure`, `feature/data-standardization`, `feature/ensemble` |

---

## Golden rules

1. **Never push directly to `main`** — always use a branch + PR.
2. **Pull `main` before creating a new branch** — start from latest code.
3. **Commit often** — small commits are easier to review and revert.
4. **Write clear commit messages** — your future self will thank you.
5. **Review PRs within 24 hours** — don't block your teammates.
6. **Delete branches after merging** — keeps the repo clean.
7. **Never commit `.env`** — it contains API keys (already in `.gitignore`).
8. **Never commit `data/`** — large files don't belong in Git (already in `.gitignore`).

---

*Last updated: Week 2*
