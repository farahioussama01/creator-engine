# Disclosure, likeness and licences

An AI-creator business that wants to reach millions has to survive a platform review and a regulator
reading its feed. The engine makes the safe path the default path.

## Disclosure is not optional in the code

- Every character has a non-empty `disclosure`; the config refuses to load without one.
- `publish.build_caption` appends it to every caption and refuses to build a caption without it.
- Every keeper gets a `.provenance.json` sidecar: IPTC digital source type
  `trainedAlgorithmicMedia`, the models, LoRA, prompts, seeds and GPU that produced it. That is the
  record that answers "how was this made" months later.
- Next step, not built: sign outputs with C2PA content credentials so platforms can read the label
  from the file itself.

## EU AI Act, Article 50

The transparency obligations apply from 2 August 2026. Deployers who publish image or video "deep
fakes" (realistic generated content that would falsely appear authentic) must disclose that it is
artificially generated (Art. 50(4)). The
machine-readable marking duty on providers of generation systems already on the market before that
date has a grace period to 2 December 2026 under the Digital Omnibus on AI (Regulation (EU)
2026/1744), which also adds a ban on systems built to generate non-consensual intimate content.
This is a summary for engineering purposes, not legal advice.

## Likeness

- Characters are `likeness: fictional` by default.
- Making content for a real creator means `likeness: licensed` plus a `consent_record`; the config
  refuses a licensed likeness without one.
- No prompts that name real people. The `new-format` skill tells Claude the same.

## Platform rules

- Publishing uses the official Instagram Graph API content-publishing flow (container, status poll,
  publish) and respects the per-account limit Meta documents (50-100 API posts per 24 h; the stricter
  number is the default).
- The scheduler adds per-account daily caps and minimum gaps. There is nothing in this repo for
  account creation, browser fingerprinting, or evading platform detection, and there won't be.

## Model licences (check the model card before commercial use)

| Component | Licence |
|---|---|
| Wan 2.2 (I2V 14B) | Apache-2.0 |
| Z-Image Turbo / Z-Image | Apache-2.0 |
| AuraFace (identity embeddings) | Apache-2.0 |
| YuNet (face detection) | MIT |
| RapidOCR | Apache-2.0 |
| InsightFace model packs | **non-commercial research only**, which is why they are not used here |
| FLUX.2 [dev] | non-commercial licence; needs a BFL commercial licence |
