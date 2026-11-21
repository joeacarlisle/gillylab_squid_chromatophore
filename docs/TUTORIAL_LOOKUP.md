# Tutorial: getting the best results with the lookup method (dashboard)

This tutorial uses the **Datasets** tab to run CHROMAS with the **lookup** segmentation and to judge and improve the
result. The running example is the CHROMAS tutorial video `example.mp4` (1000 x 1000 pixels, 20 fps, three in-focus
chunks). The numbers quoted for it show what a good and a poor result look like; they are not targets for other videos.

The `lookup` method needs no network weights and no GPU. It is the one you can improve yourself, because the result
depends only on the colors you choose.

## 1. What lookup does, and where it fails

- Every pixel color (R, G, B, each divided by 16) is looked up in a 16 x 16 x 16 table that says **background** or
  **chromatophore**. The color picker builds the table from your clicks: a pixel takes the class of the nearest clicked
  color (nearest in Lab, which follows how different two colors look).
- **One table is used for the whole video, every chunk.** A frame that is not typical gives a table that suits some chunks
  and not others.
- Lookup separates chromatophores from skin by **color only**. It cannot split two touching chromatophores by shape. If
  the soft purple halo around each pigment spot is classed as chromatophore, neighboring spots join into one territory
  with several chromatophores in it. This is the usual failure, and most of this tutorial is about avoiding it.
- It works best when chromatophores are clearly darker than the skin and at least about 10 pixels across when expanded
  (the CHROMAS paper's recommendation, eLife 2025, https://elifesciences.org/articles/106509). Overlapping chromatophores
  cannot be resolved.

## 2. Set up the run (Datasets tab)

1. Put the video in the CHROMAS `data` folder, or in `videos/` or `examples/<name>/`.
2. In **Run CHROMAS on a video**: type the file name in **Video**; give a **Dataset name** (see the tip below);
   set **Segmentation** to `lookup`; leave **Focus threshold** at 0.01 and **Use GPU** off.
3. Open the color picker card and press **Load video**.

**Tip: use a new dataset name for every attempt** (`example_v1`, `example_v2`, ...). The old result stays in the table, so
you can compare attempts row by row. The saved colors are kept per name.

## 3. Choose the frame

Use the **Frame time (s)** slider. The best frame is one where the chromatophores are **expanded and sharp**: pigment spots
at their largest, with clear edges, so that pigment and skin colors are as different as they get.

- Avoid black frames, blurred frames, and frames where the chromatophores are contracted to small dots.
- For `example.mp4`: 1 to 8 s (frames 20 to 150) is best. About 9.5 to 19 s is black, and 30 to 39 s has very few
  expanded chromatophores.
- Move the slider through a few frames before you commit. The picker builds the table from the colors, wherever they
  were chosen, so a frame that is not extreme is safer than the single sharpest one.
- If you cannot tell, run once with a reasonable frame, then open **Chromatophores > Quality checks > Focus** and
  **Pigment coverage**. They show which times were sharp and had the most pigment; pick the next frame there.

## 4. Get a first set of colors

1. Press **Suggest colours**. The frame's colors are sorted into groups; the biggest group is the background; groups close
   to it count as more background; the rest are chromatophore. The first press is reproducible. **Next suggestion** tries
   another start and **Previous suggestion** goes back.
2. Two sliders control the suggestion:
   - **Colour groups** (4 to 14): more groups find finer colors, so pigment and halo can be told apart.
   - **Background tolerance** (4 to 30): how close to the background a group must be to count as background. A higher
     value turns the halo and soft edges into background, so territories shrink to the pigment. Too high and the pigment
     itself is lost.
3. **Starting values:** the defaults (8 groups, tolerance 14) mark too much on dense skin. Start with
   **Colour groups 10 and Background tolerance 30**, then adjust by what the preview shows.

## 5. Judge the preview (the key step)

The picture on the right shows the result. Switch **View** between `Chromatophores (class > 0)`, `Class 0 outline (as
CHROMAS shows it)` and `Segmented`. The line under the picture says what share of the frame is chromatophore.

Check three things:

1. **The share matches the pigment you can see.** In `example.mp4` the dark pigment is about 15 to 20% of a frame. A table
   that marks 30 to 40% is taking in the halo (the default did, with 41% on frame 100). The picker warns when the share is
   far too low or above 60%.
2. **Neighbors are separate.** Zoom in on the densest area. Each pigment spot should have its own outline. Spots joined by
   a bridge of marked pixels will become one territory.
3. **The small chromatophores are still there.** If raising the tolerance makes the small spots disappear, go back one
   step.

**Fix by clicking**, on the left picture, after choosing the kind in the click selector:
- **Background** where chromatophore color has spread into the skin, for example on the halo between two spots.
- **Chromatophore** on pigment that is missing, in its **dark center, not its soft edge**.
- **Undo last click**, **Remove ticked** and **Clear all** correct mistakes. Every Background click counts as class 0 and
  every Chromatophore click as class 1, so the background can have as many shades as it needs.

Then move the frame slider to a frame from each chunk (for `example.mp4`, around 3 s, 21 s and 43 s) and check the share
and the separation again. The same table is used for all of them.

## 6. Run, and read the result

Press **Use these colours and run CHROMAS**. The output box shows progress. When it finishes the dataset appears in the table.

Read its row:

| Column | Good | Warning sign |
|---|---|---|
| **measured / numbered** | 65 to 75% is what the project's two squid videos (FASTEC, TS-2022) reached | Below about 30%: merged or oversized territories |
| **measured per chunk** | Similar in every chunk | One chunk far below the others |
| **median area per chunk (px)** | Similar in every chunk, near the size of one chromatophore (150 to 250 px for the example) | One chunk several times larger: merged chromatophores |
| **chunks agree** | `yes` | `NO: numbered ...`: the slice step will fail |

Then check the pictures in the **Chromatophores** tab: **Chromatophore territories** (each region should hold one
chromatophore), **Segmentation check**, and **Make video** (`segmentation` or `territories`) in the Datasets tab.

## 7. Improve the result, one change at a time

Change one thing, run under a new name, and compare the rows.

1. **Colors first.** If a chunk has a large median area or few measured, redo the picker: a different frame, more groups,
   or a higher tolerance, as in section 5.
2. **Then the area limits.** Open the Datasets tab, tick the dataset, set **Minimum territory area (px)** and **Border to
   remove (px)**, and press **Measure areas again**. The defaults are 200 and 40. In a check on chunk 0 of the example the border
   was the setting that changed the count (a smaller border keeps territories near the frame edge); the minimum area did not. This
   step is fast because it does not segment again, but it renumbers the territories, so run the slice, grouping and
   motor-unit steps again afterward. If a whole part of the picture has no territories (for example separate patches far from the main group of chromatophores), raise **Mask widening** (default 10; try 30 to 40) in the same row: CHROMAS gives territories only inside the largest connected group of pigmented regions, and a larger value joins the far patches to it.
3. **Reuse good colors.** The picker keeps the table of a run. A plain **Run CHROMAS** with lookup on the same video reuses them.
   To make a new table, press **Suggest colours** or click again in the picker.

## 8. Example of reviewing three attempts

The numbers below are made up to show how to read the rows; they are not from a real video. Each attempt is a new dataset
name, and the **Review** column is what you would write down after reading the row (section 6) and looking at the territory
picture.

| Attempt | Colors | Review |
|---|---|---|
| `video_v1` | Defaults: 8 groups, tolerance 14 | 40% measured of numbered. One chunk has a median area five times the others: its territories hold several chromatophores. The counts per chunk differ. **Too much halo marked.** |
| `video_v2` | 10 groups, tolerance 30, a different frame | 62% measured. The median areas are similar in all chunks. The counts still differ by one, so `chunks agree` shows `NO`. **Merging is fixed; one chunk still numbers differently.** |
| `video_v3` | As `video_v2`, plus a few Background clicks on the halo between touching chromatophores | 70% measured, similar in every chunk, `chunks agree` = `yes`. Territories hold one chromatophore each. **Good enough to go on:** adjust only the area limits next. |

How the review leads to the next attempt:
- Large median area in one chunk: raise the tolerance, add Background clicks on the halo, or use a frame from that chunk.
- Few measured in one chunk: its chromatophores are too light or too small for the table; click a few of them as Chromatophore, or try a different frame.
- `chunks agree` = `NO`: change the colors, not the area limits, and run again.
- Rows that look right but territories are missing in one part of the picture: see **Mask widening** in section 7.

## 9. Checklist

- [ ] Frame: expanded, sharp, typical; checked on a frame from each chunk.
- [ ] Share marked is close to the pigment you can see; not above 40% on dense skin.
- [ ] Neighbors separate in the densest area; small chromatophores kept.
- [ ] New dataset name for each attempt.
- [ ] Row: measured per chunk and median area per chunk similar in all chunks; `chunks agree` = `yes`.
- [ ] Area limits adjusted only after the colors are right.
