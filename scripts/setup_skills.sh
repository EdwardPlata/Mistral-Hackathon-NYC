#!/usr/bin/env bash
set -euo pipefail

# Install skills CLI (skills.sh / vercel-labs)
if ! command -v skills &> /dev/null; then
  npm install -g @vercel/skills || npx skills --help || true
fi

# Clone Hugging Face Skills
if [ ! -d "skills/hf" ]; then
  mkdir -p skills
  git clone https://github.com/huggingface/skills.git skills/hf
fi

# Clone Awesome Claude Skills
if [ ! -d "skills/awesome-claude-skills" ]; then
  git clone https://github.com/BehiSecc/awesome-claude-skills.git skills/awesome-claude-skills
fi

# Clone Skill_Seekers
if [ ! -d "skills/Skill_Seekers" ]; then
  git clone https://github.com/yusufkaraaslan/Skill_Seekers.git skills/Skill_Seekers
fi

echo "Skills repositories installed under skills/."
