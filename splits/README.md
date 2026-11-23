# The paper's source-grouped split

`paper_grouped_split.csv` lists every clip of the paper's detection corpus with its partition:
- **`split`:** train, val or test;
- **`label`:** drone or non_drone;
- **`file`:** the clip's file name after segmentation and mixing;
- **`source_group`:** the source recording.

All clips of one source group are in one partition, and no group crosses partitions.

**Counts (clips):**

| Partition | Drone | Non-drone |
|---|---|---|
| Train | 2,708 | 7,366 |
| Validation | 583 | 1,578 |
| Test | 585 | 1,578 |

**What the training set adds:** 139 synthetic drone clips, generated at training time and not listed.

**Left out:** 13 training clips cut from a private recording (see `MODEL_CARD.md`).

**Source groups:**
- `dads:*`: DroneAudioDataset recordings; its ESC-50 and Speech Commands noise slices are `esc50:*` and `sc:*`.
- `us8k:*`: UrbanSound8K recordings.
- `video:*`: public drone videos.
- `silence`: generated silence.
- `mixdrone_*` files: a drone clip mixed with a background; they inherit the group of their drone source.

The clips themselves are not distributed; get the datasets from their owners under their licences.
The group of a file name can be recomputed with the rules in the paper (Section III-A).
