---
name: new-format
description: Draft a new content format YAML (keyframe prompt, motion prompt, scene/outfit/light pools, hooks) for a character, ready for the experiment loop.
---

Create `configs/formats/<id>.yaml` for: $ARGUMENTS

Follow the existing files in `configs/formats/` exactly (same keys). Requirements:
- `keyframe_prompt` uses `{trigger}`, `{look}`, `{scene}`, `{outfit}`, `{light}`; phone-photo realism cues
  ("shot on a phone", "realistic skin texture"), no celebrity names, no real brands unless the user
  asked and listed them in `allowed_text`.
- `motion_prompt` describes one simple motion that keeps the face toward camera (head turns and
  hands near the face are the top causes of identity morph and finger artifacts in Wan clips).
- At least 4 scenes, 3 outfits, 2 lights, 3 hooks. Hooks are the first line of the caption.
- `head_ratio` matches the shot: full body ≈ [0.06, 0.20], medium ≈ [0.10, 0.28], close-up ≈ [0.25, 0.55].
- If text appears in frame, set `has_signage: true` and put the exact words in quotes in the prompt.
- Set `for_characters`.

Then plan a `draft` batch of 6 for it with `plan_batch` (quotas `{"<id>": 6}`) so it enters the
experiment loop with exploration slots. Do not run it without asking.
