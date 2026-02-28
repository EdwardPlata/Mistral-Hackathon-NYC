---
name: safe-reader
description: "Read-only exploration mode with no file modifications or command execution"
allowed-tools: ["read_file", "grep_search", "semantic_search", "file_search", "list_dir", "get_errors"]
---

# Safe Reader Mode

Explore the repository without any risk of modifications or side effects.

## Purpose
A guaranteed read-only mode for:
- Initial exploration when requirements are unclear
- Code review and analysis
- Answering questions about existing code
- Understanding architecture before making changes
- Safe browsing when uncertain

## Capabilities
You can:
- ✅ Read any file in the repository
- ✅ Search for code patterns (grep, semantic search)
- ✅ Find files by name or pattern
- ✅ List directory contents
- ✅ Check for compilation/lint errors
- ✅ Analyze code structure and dependencies

## Restrictions
You cannot:
- ❌ Modify any files
- ❌ Create new files
- ❌ Run commands or scripts
- ❌ Execute tests
- ❌ Install packages
- ❌ Make git operations

## When to Use

### Use Safe Reader when:
- User is asking "how does X work?"
- Exploring unfamiliar code
- Reviewing changes before making decisions
- Clarifying requirements
- Searching for patterns or examples
- Security-sensitive exploration

### Exit Safe Reader when:
- Requirements are clear and changes are needed
- User explicitly requests modifications
- Need to run tests or validation
- Ready to implement features

## Workflow
1. **Understand the question**: What does the user want to know?
2. **Plan exploration**: What files/modules are relevant?
3. **Gather context**: Read, search, and analyze
4. **Synthesize findings**: Provide clear, structured answers
5. **Suggest next steps**: Recommend actions if changes are needed

## Output Format
Structure your response as:

```
## Finding: [Brief summary]

### Context
- File: path/to/file.py
- Lines: 45-67
- Purpose: [what this code does]

### Analysis
[Detailed explanation]

### Relevant Code
[Show key snippets with context]

### Next Steps (Optional)
If changes are needed: [suggest moving to a different skill]
```

## Usage
Invoke with: `/safe-reader`

This mode stays active until you explicitly exit or invoke another skill.

## Transitioning to Action
When ready to make changes, suggest the appropriate skill:

- Need to refactor? → `/refactor-with-tests`
- Need to run tests? → `/run-tests`
- Need to update docs? → `/doc-sweep`
- Need deep analysis? → `/deep-research`
- Ready to commit? → `/commit`

## Philosophy
"First understand, then act." Safe Reader enforces thoughtful exploration before making potentially disruptive changes.
