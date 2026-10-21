# NoobAI reference images

Inputs used while validating the two-character composition pipeline. They are
checked in so the same run can be reproduced on a fresh instance.

| file | size | role |
|---|---|---|
| `doll_pose_reference.png` | 1024x1280 | Body Chan ball-joint doll photo. Pose source. |
| `castle_room_reference.png` | 896x1152 | Castle bedroom reference, cropped to portrait 3:4. |

## Why the doll is useful

The doll gives an exact two-figure pose: who holds whom, arm angles, leg
positions, hand and finger placement. It is a pose reference, never a look
reference — joints and plastic must not reach the output.

## How the doll is turned into conditioning

DWPose does **not** work on the doll. Its person detector is trained on humans
and rejects a featureless plastic mannequin, so the skeleton comes out as
fragments — one head, one arm, no torso and no second figure.

Line-art extraction does work, because it reads geometry only:

```
LoadImage(doll_pose_reference.png)
  -> LineArtPreprocessor(coarse=enable, resolution=1024)
  -> ControlNetApplyAdvanced(union, type=anime_lineart, strength 0.85-1.0)
```

`coarse=enable` matters: at default settings the doll's ball-joint rings are
picked up as ink and reappear as seams on the generated skin.

## Why the room crop exists

The room reference is 4:3 landscape while the pose line-art is portrait. Mixing
them on one canvas makes ControlNet stretch the figures across the wide frame,
and the composition collapses.

The fix applied here, and the rule to keep:

1. Crop or pad every reference to the **same aspect ratio** as the output canvas.
2. Composite the pose line-art onto that canvas at 70-80% height, lower-centre,
   on black. Letterboxing is what keeps figures inside the room instead of
   filling it.
3. Feed the padded line-art to ControlNet and the matching room crop to
   `VAEEncode` for img2img.

Validated at 896x1152 with denoise 0.50 and ControlNet strength 0.85-1.0.

## Known weaknesses on this pair

- Two overlapping bodies can merge; the lower figure's face and arms get soft.
- White fabric picks up the red of the canopy — describe garment colours
  explicitly and keep denoise low.
- Hands remain the weakest area. In-place inpainting does not fix them: the hand
  is roughly 52x39 px on this canvas. Crop the forearm together with the hand,
  upscale about 4x, repaint, then paste back with a feathered mask.
