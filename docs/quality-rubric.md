# Good, hype, brain rot: what the gates look at

**Good** passes as a real phone photo or clip to someone scrolling past. **Hype** is technically clean
but reads as AI gloss: airbrushed skin, impossible studio light in a kitchen, everything centred.
**Brain rot** has artifacts people screenshot: six fingers, a head too big for the shoulders, a sign
that says "COFEE", a face that turns into someone else halfway through the clip.

| What a reviewer checks | Typical failure | Automated check | Stage |
|---|---|---|---|
| **Light** | blown highlights, crushed shadows, murky flat exposure; light that disagrees with the scene | `exposure`: clipped-pixel fractions and mean luminance; Claude `light` score | keyframe |
| **Skin** | waxy, plastic, pore-less skin | `skin_texture`: robust high-frequency energy (MAD of a high-pass) inside the face, normalised for exposure, measured near native resolution so face size doesn't matter; Claude `skin` score | keyframe |
| **Hands** | extra or fused fingers, hands melting into props | Claude `hands` score (pixel statistics cannot see this); formats whose motion keeps hands away from the face | keyframe, clip |
| **Head size** | the AI "big head" | `head_size`: face height ÷ frame height against the format's shot range (full body ~0.07-0.20); Claude `anatomy` score | keyframe |
| **Sign text** | near-miss spelling, letter soup | `sign_text`: OCR, then each word is checked against the quoted words in the prompt and a signage wordlist; one or two edits away from a known word fails | keyframe |
| **Identity** | the face drifts from the character | `identity`: AuraFace embedding cosine to the character centroid, threshold suggested at enrollment | keyframe |
| **Motion** | frozen clip, chaotic camera, flicker, face morphing mid-shot | `flicker`, `motion`, `temporal_identity` (identity on 8 frames across the clip, worst frame must pass); Claude `motion` score | clip |
| **Edit style** | hooks that don't stop the scroll, wrong pacing | not a pixel property: formats carry hooks and pacing, and the experiment loop scores them by views vs account baseline and skip rate | posts |

## Why the order matters

Gates run cheapest first and stop at the first failure. Pixel statistics cost microseconds, face
embeddings milliseconds, OCR a few hundred milliseconds, a Claude review a fraction of a cent. The
Claude judge only ever sees survivors, which is why judging every keeper costs less than a second of
GPU time.

## Calibrating on real renders

Simulator thresholds are not real thresholds. On the first real batch:

1. Label 40-100 renders: `file,label,issue` where label is `good`/`bad` and issue optionally names the
   gate that should have caught it (`identity`, `skin_texture`, ...).
2. `ce gates calibrate <batch> --labels labels.csv` re-runs every gate without short-circuiting and
   prints, per gate, recall and false-reject rate at the current threshold and at the threshold that
   maximises both.
3. Change one threshold at a time, re-run on the next batch, compare funnels in `ce report`.

On the simulator, where the injected defects are known, the same command already caught one thing:
the clip identity threshold of 0.62 missed half the morphs; 0.70 catches them with no false rejects.

## Known blind spots

- The skin texture measure can be fooled by heavy sensor noise or JPEG artifacts; calibrate it on
  outputs from the exact pipeline, after any upscaler.
- Head-size ratios assume one person in frame. Two-person formats need their own range.
- OCR misses stylised neon or cursive signage; the Claude `text` score is the backstop.
- Identity on profile views is weaker for every face embedder; formats with lots of profile shots need
  a lower threshold or reference images in profile at enrollment.
