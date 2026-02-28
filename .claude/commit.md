---
name: commit
description: "Safe git commit workflow with staged changes and confirmation"
disable-model-invocation: true
allowed-tools: ["run_in_terminal", "read_file", "grep_search"]
---

# Commit Changes

A safe, deliberate git commit workflow that requires user confirmation.

## Workflow
This is a **manual-only skill** (prevents accidental commits).

### Phase 1: Review Changes
1. Check git status: `git status -sb`
2. Show unstaged changes: `git diff`
3. Show staged changes: `git diff --cached`
4. Summarize what files changed and why

### Phase 2: Stage Changes
If changes aren't staged yet:
```bash
git add <files>
```

Staging strategies:
- **Specific files**: `git add file1.py file2.py` (preferred for focused commits)
- **All changes**: `git add -A` (use when all changes are related)
- **Interactive**: Suggest `git add -p` for partial file staging

### Phase 3: Craft Commit Message
Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Examples**:
```
feat(agents): add tool validation for LLM calls

- Implement schema validation for tool parameters
- Add error handling for malformed tool responses
- Update tests to cover validation edge cases

Closes #42
```

```
fix(ml): resolve GPU memory leak in training loop

- Clear cache after each batch
- Reduce batch size for large models
- Add memory profiling in debug mode
```

**Simple example**:
```
docs: update README with UV setup instructions
```

### Phase 4: Commit
```bash
git commit -m "type(scope): subject" -m "body"
```

Or for complex messages, create commit interactively:
```bash
git commit
```

### Phase 5: Verify & Push (Optional)
1. Show the commit: `git show HEAD`
2. Check branch status: `git status -sb`
3. Ask user if they want to push: `git push origin <branch>`

## Safety Features
- **Manual-only**: Cannot be auto-invoked by the model
- **Review-first**: Always show diffs before committing
- **Confirmation**: User approves message before commit
- **No auto-push**: User decides when to push

## Rollback
If commit was wrong:
```bash
# Undo commit, keep changes staged
git reset --soft HEAD~1

# Undo commit, keep changes unstaged
git reset HEAD~1

# Undo commit, discard changes (CAREFUL!)
git reset --hard HEAD~1
```

## Usage
Invoke with: `/commit [optional_message_type]`

Examples:
```
/commit
/commit feat
/commit fix
```

## Integration
- Reads changed files to understand context
- Suggests appropriate commit type and scope
- Can reference issue numbers from git branch name
- Works with conventional commits for changelog generation
