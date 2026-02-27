# Skills Overview

This project uses the following skill sources:

- **skills.sh marketplace** via `skills` CLI ([vercel-labs/skills](https://github.com/vercel-labs/skills))
- **Hugging Face Skills** ([huggingface/skills](https://github.com/huggingface/skills))
- **Awesome Claude Skills** ([BehiSecc/awesome-claude-skills](https://github.com/BehiSecc/awesome-claude-skills))
- **Skill_Seekers** ([yusufkaraaslan/Skill_Seekers](https://github.com/yusufkaraaslan/Skill_Seekers)) for generating custom skills from vendor docs

See [`skills-plan.md`](./skills-plan.md) for which skills to enable first.

## Quick Install

Run the helper script to clone all skill repositories into `skills/`:

```bash
bash scripts/setup_skills.sh
```

## Vendor-Specific Skills

Custom skills for NVIDIA NeMo Agent Toolkit, ElevenLabs, and W&B can be
generated with Skill_Seekers and stored under `skills/vendor/`.
