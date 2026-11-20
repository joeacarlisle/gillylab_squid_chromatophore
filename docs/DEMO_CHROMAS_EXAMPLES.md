# Demo from the dashboard: the CHROMAS example videos

This walks you through one complete run in the dashboard, with no command line: the CHROMAS authors' example video
(close-up skin with dark chromatophores on a light background, 1000 x 1000 pixels, 20 frames/s) goes in; the area of every
chromatophore, its position, and the activation analysis come out. You use the **Datasets**, **Chromatophores**, **Activation** and (optionally) **Advanced analytics** tabs.

```
video  --CHROMAS (Datasets tab)-->  area of every chromatophore in every frame, and its position
       --Activation tab-->  activation events  ->  are neighbours coupled? (p value)  ->  waves (speed, direction)
```

The video is the whole 1000-frame recording. According to the authors' tutorial it has five phases of about 10 seconds
each: in focus, too dark, in focus but rotated, too blurry, in focus. CHROMAS first cuts it into chunks: it finds the three in-focus stretches
itself, segments each one, and joins the results into one dataset. In my run the three chunks had 198, 199 and 199 frames
(about 10 s each at 20 frames/s, at 0 to 10 s, 20 to 30 s and 40 to 50 s of the video), so the plots have gaps where the dark
and blurry phases were left out.

Needs no GPU. Time: the colour (lookup) run of `example.mp4` took roughly 12 minutes on one CPU core of a laptop. The
dashboard now uses 2 cores by default (`n_workers` in `config.defaults.toml`); if a run stops with an *out of memory*
message, set it to 1. The neural-network route takes a few hours (an estimate, not measured on this video).

The example data is not part of this repository. Step 0 gets it. (An optional check on artificial data, which needs no video, is at the end.)

---

## Step 0. Get the example data and check the setup

1. Both projects are installed, side by side: `gillylab_chromas` (see its README) and this folder (see the README,
   "Setup"). The dashboard starts with `scripts\windows\chromatophore_dashboard.bat`, which opens
   http://localhost:5009 on the **Datasets** tab (the page title is Gilly Lab - CHROMAS).
2. Download `Chromas2025.zip` from https://doi.org/10.17617/3.MJI3HA (check the licence on that page first), put it in
   the `gillylab_chromas` folder, and run `build_demo.bat` there. It copies the example videos into
   `gillylab_chromas\data\`:

   | File | What it is |
   |---|---|
   | `example_chunk_0.mp4` | 201 frames (10 s). The first of the three in-focus stretches of `example.mp4` (a quick test: it runs in a third of the time) |
   | `example_chunk_1.mp4`, `example_chunk_2.mp4` | the other two stretches (211 and 200 frames) |
   | `example.mp4` | the whole recording, 1000 frames (50 s). **Used below.** CHROMAS finds the in-focus stretches itself |

3. Open the **Datasets** tab. It should say *CHROMAS found in ...* and *Videos go in ...\data*. If a yellow *CHROMAS is
   not set up* box shows instead, fix what it lists first.

---

## Step 1. Run CHROMAS on the example video (it cuts it into chunks)

1. Open the **Datasets** tab and find **Run CHROMAS on a video**.
2. Fill in the form:

   | Field | Value |
   |---|---|
   | Video (file name or full path) | `example` (the `.mp4` may be left off; it is looked for in the CHROMAS data folder) |
   | Dataset name | `chromas_example` |
   | Segmentation | `lookup` |
   | Focus threshold | `0.01` |
   | Use GPU | off |

3. Open the card **Choose the lookup colours here, with a live preview**. The lookup method labels pixels by colour,
   and this card chooses the colours from the video itself:
   1. Press **Load video**. It shows a frame and suggests colours from it: the biggest colour group (the light skin)
      is the background, and the other groups (the dark chromatophores) are chromatophore. The picture on the right
      shows the result, with outlines round what will be segmented.
   2. Check it. On the first frame the dark chromatophores should be outlined cleanly; the small orange dots and the
      soft, out-of-focus ones may be outlined too, which is fine for a first run. To correct it, click on the frame
      on the left with **Background** or **Chromatophore** selected, in the middle of the colour you mean, not on its
      edge. **Undo last click**, **Remove ticked** and **Clear all** undo clicks; **Suggest colours** starts again.
   3. Move **Frame time (s)** to look at a later frame before you run.
4. Press **Use these colours and run CHROMAS**. The table of colours is saved with the dataset, and the run starts.
   The output box shows CHROMAS's progress (focus check, segmentation, registration, area).
5. Wait for *Done: dataset "chromas_example" is loaded*. The dataset now appears in the sidebar and in the
   **Chromatophores** tab, which shows the map of chromatophores, the area of each over time, and a heat map.

At the end of the run the output box may show *Slicing failed! conflicting dimension sizes*. The run itself has completed (the
line before says *Pipeline run completed successfully*): only the optional slice step, which the run tries last, could not
join the chunks. Step 6 explains and fixes it. The Dask message *Cannot synchronously wait on a running event loop* when
CHROMAS shuts down is harmless.

What a run of `example.mp4` gives (colour lookup, one core): CHROMAS numbers several hundred chromatophores, and only
some of them get a measured area (the rest are left out, for example because they sit at the edge or are not found in
every frame). The exact numbers change from run to run, with the colours you click, the focus threshold and the
CHROMAS version, so do not expect to match anyone else's counts. The Datasets tab shows the counts for your dataset.

The bridge watches the result: if more than 75 % of the frame is labelled chromatophore it stops the run and says so.
(In my test on the first frame of `example_chunk_0`, the suggested colours labelled about 40 % of the frame. The whole recording also contains out-of-focus stretches: if the frame you load looks blurred, move **Frame time (s)** to a sharp one before you choose colours.)

**How the colours affect the result.** In a test on frame 0 of `example.mp4`, the suggested colours gave a mask that follows
the large dark spots and the small brown spots closely (about 40 % of the frame). The soft purple halo round the spots is
counted as chromatophore, so outlines run a little wide and some neighbours join. Treating that halo as background in the
colour picker (it allows several Background clicks) gave a tighter mask, about 32 % of the frame, with far fewer large
merged blobs (5 blobs over 3,000 pixels against 28). Which one is more accurate is not known: there were no hand-made masks
to score them against. If you use CHROMAS's own clicking window instead, only the first click is background and every
other click is its own class that counts as chromatophore. In a simulation of that window, three clicks (one pale
background, one dark spot, one small brown spot) gave about 34 % of the frame, while five clicks gave about 55 % and merged
neighbours, so use few clicks there. I did not run CHROMAS's window itself.

**Trying several colour suggestions.** In the colour picker, **Suggest colours** gives the same suggestion every time for the same frame, groups and tolerance. **Next suggestion** tries another random start and keeps the earlier ones; **Previous suggestion** and **Next suggestion** step back and forth between them, and the label says which one is shown (for example *Suggestion 3 of 5*). Clicks you add or remove on a suggestion are kept when you move away and come back. Pick the one whose preview looks best, then press **Use these colours and run CHROMAS**. **Suggest colours** starts a new list.

**Running the same video again.** The colour table is saved for the video (`output/<name>.lookup_table.npz`) and reused
by the **Run CHROMAS** button, so a second run does not open a clicking window and gives the same segmentation (the first
run has no saved colours, so CHROMAS asks for the clicks). To click new colours in CHROMAS's own window, tick **Click new
colours (lookup)**. In the colour picker card, a box names the saved table and says when it was made, and **Run with the
saved colours** starts CHROMAS with it; **Use these colours and run CHROMAS** replaces it with the colours you chose. A copy
of the colours behind each lookup result is kept with it, in `processed/chromas_export/<dataset>/lookup_table.npz`. A run
with the same dataset name replaces that dataset, including any slice, grouping or motor-unit results you added to it, so
run those steps again afterwards, and press the import button again to refresh the other tabs. To keep the old result,
give the new run another name. To test another minimum territory size without a new run, use **Measure areas again** in
the Datasets tab.

**Neural network instead of colours.** Choose `neuralnet` as *Segmentation* and leave *Model file* empty. It uses the
trained weights that `build_demo.bat` put in `gillylab_chromas\model_weights\` (the authors trained them on cuttlefish and
bobtail squid skin), and it needs no clicking. On a CPU expect a few hours for this video (not measured).

Command line:

```
python chromas_bridge.py run example --method lookup --name chromas_example
```

(The command line opens CHROMAS's own colour-clicking window; click one patch of light skin first, then one dark
chromatophore, then press Q.)

---

## Step 2. Look at the chromatophores (Chromatophores tab)

Open the **Chromatophores** tab and choose `chromas_example` in the sidebar. Do this before the activation analysis: it
shows whether CHROMAS found the chromatophores correctly, and everything later depends on that.

1. **Map.** One dot per chromatophore, at its position on the skin, coloured by a measure you can change. Check that the
   dots sit on the dark chromatophores and that none are missing from a region that has them.
2. **Chromatophore areas.** Choose one or more chromatophores in the list. The plot shows the area of each over time
   (tick *relative* to see it as a fraction of its own range). Chromatophores expand and relax in bursts, so you should see
   a flat resting level with spikes up. A trace that is flat everywhere, or noisy with no pattern, usually means a bad
   segmentation (go back to Step 1).

   For `example.mp4` the plot has three stretches, at 0 to 10 s, 20 to 30 s and 40 to 50 s: chunks 0, 1 and 2. The gaps are
   the dark and blurry phases that CHROMAS left out. Chunk 1 is the rotated stretch. In some runs its chromatophore
   numbers do not match those of the other two chunks (CHROMAS may find fewer territories there, and a number may point to
   a different chromatophore), and then a trace can jump in the middle stretch. The Datasets tab lists each chunk's
   chromatophore count: when a chunk's count differs from the others, treat that stretch with caution.
3. The panels below are collapsed. Each is drawn the first time you open it:
   * **All chromatophores (heat map)**: every chromatophore against time. Vertical stripes are moments when many expand
     together.
   * **Sizes and activity**: histograms of resting size, largest expansion and time spent expanded.
   * **Groups that move together**, and **Activity per chromatophore and territories**.
   * **Slices (flower plots)**: empty until the CHROMAS slice step has been run (see Step 6).
4. **Quality checks** (also collapsed): **Focus**, **Pigment coverage**, **Registration between chunks**,
   **Segmentation check** and **Overlay video**. With `example.mp4` the dataset has three chunks, so *Registration between
   chunks* is meaningful here: the chromatophores of neighbouring chunks should line up. The overlay video is made in the
   **Datasets** tab (*Make video*, with the dataset ticked) and draws what CHROMAS found on the footage.

---

## Step 3. Analyse it (under a minute)

1. Open the **Activation** tab and press **Refresh**.
2. *CHROMAS dataset*: choose `chromas_example`.
3. Leave *Settings file* empty. *Pixels per mm* stays 0, so speeds come out in pixels per second. *Region* stays empty,
   so the whole frame is analysed. Leave the thresholds as they are for a first look.
4. Press **Analyse dataset** and wait for the green *Done* box. The results appear below.

Command line:

```
python chromas_activation.py analyse chromas_example
```

This video has 201 frames (10 s), so there may be few activation events, and "no evidence that neighbours are more
coupled" is a possible and valid answer. It shows that the pipeline runs from video to result; it is not a finding
about the skin in the video.

---

## Step 4. Read the result

Work down the page in this order.

**1. Result.** One paragraph in words: how many chromatophores were measured, how many activation events were
found, whether neighbours are more coupled than distant pairs (with the p value), and how many waves there were. A
green box means the coupling test came out significant (p < 0.05).

**2. Are neighbours activated together?** The table has one row per measure:

| Column | Meaning |
|---|---|
| `rate_correlation` | how alike the expansion rates of two chromatophores are, over the whole recording (no event threshold involved) |
| `event_coincidence` | the share of their activations that start within 0.2 s of each other |
| `neighbour_mean` / `other_mean` | the average for neighbouring pairs, and for all other pairs |
| `p_value` | how often shuffling the chromatophores around the skin gives a neighbour advantage this large (999 shuffles) |

The plot below it shows coupling against distance. If coupling is local, both lines are highest at about 1 spacing
and fall away; if they are flat, chromatophores far apart are as alike as neighbours (for example, because the
whole patch responds to the stimulus at once).

**3. Activation events.** Every dot is one activation onset; the vertical axis orders chromatophores from left to
right in the frame. A slanted line of coloured dots is a wave moving across the skin; a vertical line is many
chromatophores activating at the same moment; grey dots are events that belong to no wave.

**4. Traces.** Pick chromatophores in the list (most events first) to see their relative area (0 = at rest, 0.3 =
30 % larger) with the detected onsets as dotted lines. **Use this to judge the detection**: every clear expansion
should have a dotted line at its start, and flat stretches should have none. If not, see step 5.

**5. Waves.** One row per wave:

| Column | Meaning |
|---|---|
| `kind` | `travelling` (onsets follow a direction across the skin), `simultaneous` (all within one frame) or `unstructured` |
| `speed_mm_per_s`, `direction_deg` | for travelling waves; direction in the image: 0 = right, 90 = down, 180 = left, -90 = up |
| `r2`, `p_r2` | how well onset time follows a straight front, and whether that is more than chance |
| `first_chromatophore` | where the wave started |

Choose a wave under *Activation map* to see it drawn on the skin: colour is onset time (blue first, red last), small
arrows point from the neighbour that activated just before, and the black arrow is the fitted direction of travel.

A group of linked activations only counts as a wave if it is larger than what the same events produce by chance;
the result box says how large such chance groups get for this recording.

---

## Step 5. Adjust the settings (if needed)

Change a value in the form and press **Analyse dataset** again; the results are replaced.

| What you see | Change |
|---|---|
| Clear expansions in the traces have no dotted line | lower *Onset (noise sigmas)* (4 -> 3) or *Min. growth* (0.10 -> 0.05) |
| Dotted lines on flat, noisy stretches | raise *Onset* (4 -> 5) or *Min. growth* (0.10 -> 0.15) |
| One wave is split into several | raise *Max. lag between neighbours* (0.5 -> 0.8 s) |
| Separate events are joined into one big wave | lower *Max. lag* (0.5 -> 0.3 s) |
| Chromatophores across a gap count as neighbours | lower *Neighbour edge limit* (1.6 -> 1.3) |

Note every change you make with the result: the settings used are saved in `summary.json` next to the results.
The defaults are starting values; settle them on clips the lab has scored by hand, then keep them fixed across
recordings so the results can be compared.

---

## Step 6. Advanced analytics (optional, after the activation analysis)

The **Advanced analytics** tab has reports modelled on the figures of the CHROMAS papers. Choose `chromas_example` in the
sidebar; each report is drawn when you open its panel. Some need a CHROMAS step that you start yourself in the
**Datasets** tab: tick the dataset in the table, then press the button below it. These steps run in the background and
a bar shows progress.

| Report | What it needs |
|---|---|
| **Chromatophore groups (CHROMAS area clusters)** | Press **Run CHROMAS grouping**, with `0,2` in the **Chunks** box (it then runs once for each chunk listed; with the box empty it uses the chunks that match) |
| **Expansion against relaxation** | Nothing extra |
| **Chunk overlap** | Nothing extra; compares chunks, so it needs a dataset with two or more (this one has three) |
| **Wedge traces**, **Wedge correlation clusters**, **Cluster statistics**, **Cluster network**, **Wedge orientation** | Press **Run slice step** with `0,2` in the **Chunks** box next to the buttons (cuts each chromatophore into 36 wedges; the dashboard says minutes to an hour) |
| **Motor units** (the papers' method) | **Run slice step** first, then **Run motor-unit analysis** |

Things to know:

* **Why the Chunks box.** The slice step and the grouping join the chromatophore labels of all chunks, which only works if
  every chunk has the same number. If one chunk has a different count (in some runs of this video the rotated chunk 1
  does), using all chunks fails with *conflicting dimension sizes* (slice) or a size error (grouping). The authors' tutorial
  says `chromas slice` can be limited with `--chunk-selection` (for example `0,2`); the Chunks box passes exactly that
  (the grouping runs once per chunk listed). The reports then cover only the chunks you listed. When you tick a dataset
  in the Datasets tab, a table lists each chunk's frames and chromatophore count and says what to type; if all counts
  match, leave the box empty.
* The slice step works best when a fully expanded chromatophore is at least 50 pixels across. I have not checked this for
  the example video; if the wedge reports come out empty or odd, that may be why.
* *Wedge correlation clusters* is this project's own shortcut, not the papers' method. Do not report them as motor units;
  *Motor units* is the papers' method.
* A panel that is missing its step says so and names the button. In the motor-unit analysis the ICA step can finish while the
  clustering of the components that follows stops with *cluster_correlation could not converge after 10 attempts*. The
  dashboard then retries that step with stronger damping and, if needed, with another method (HDBSCAN, not the papers'
  default), and the log says which setting was used. Report it if you use the HDBSCAN result.

---

## Step 7. Where the results are

Everything is in `processed\chromas_activation\<dataset>\`:

| File | Content |
|---|---|
| `summary.txt`, `summary.json` | the result in words; numbers and the settings used |
| `events.csv` | one row per activation: chromatophore, onset, peak, half-relaxation time, rise time, amplitude, wave |
| `chromatophores.csv` | position, resting size (px and mm^2), events per minute, number of neighbours, why one was left out |
| `coupling_tests.csv`, `distance_profile.csv`, `neighbour_pairs.csv` | the coupling test and its parts |
| `waves.csv`, `wave_links.csv` | waves, and each link from the activating neighbour to the next chromatophore |
| `areas_wide.csv`, `centroids_wide.csv` | areas (mm^2) and positions, one column per chromatophore, like the old pipeline's spreadsheet |
| `pathways.png`, `wave_map_<n>.png` | the pictures shown in the tab |

These CSV files open in Excel.

---

## Optional: check the method on artificial data (1 minute)

This needs no data and does not use your video or the Datasets tab. The button makes two artificial datasets by itself, so you can see what a clear result looks like. Skip it if you only want the demo; do it before or after Steps 1 to 5.

1. Open the **Activation** tab.
2. Press **Check the method (synthetic data)**.
3. Wait for the green box: *Done: the method found the planted waves and no coupling in the control*.

This made two artificial patches of 180 chromatophores. In `synthetic_waves` two waves were planted (400 px/s to the
right at 4 s, 250 px/s towards the lower left at 12 s) on top of random twitches. In `synthetic_no_coupling` the same
number of events happen at random times.

4. Choose **synthetic_waves** in *CHROMAS dataset* and scroll down. You should see:
   * **Result** (green): *neighbours are more coupled than distant pairs*, p = 0.001, and 2 travelling waves.
   * **Coupling against distance**: both lines high at 1 spacing (neighbours) and falling with distance.
   * **Activation events**: two slanted bands of coloured dots. Each band is one wave: chromatophores further right
     (higher up the plot) activate later.
   * **Waves**: about 393 px/s at 1 degree and 250 px/s at 135 degrees, R^2 about 0.97.
5. Choose **synthetic_no_coupling**: grey result box, p about 0.77, no waves, and the event plot is scattered grey dots.

Keep these two pictures in mind: they are what "coupled with waves" and "independent" look like.

Command line: `python chromas_activation.py example`

---

## Your own video

1. Put your video in the CHROMAS data folder (`gillylab_chromas\data\`), or give its full path in the run form.
   The **Video** tab can crop, trim and convert it to H.264 first.
2. Follow steps 1 and 3 with your video's name. For speeds in mm/s, fill in *Pixels per mm* in the Activation tab
   (from the magnification of your recording), and give a *Region* (x y width height) to analyse only part of the frame.
3. To keep settings for a video, put them in a toml file in a new folder under `examples/` (the Activation tab's
   *Settings file* lists such folders) with an `[activation]` table; see `config.defaults.toml` for the values.

The video must be a close-up in which each chromatophore is at least about 10 pixels across; CHROMAS cannot
separate chromatophores in distant footage.

---

## When something goes wrong

| Message or symptom | What to do |
|---|---|
| *... was imported before chromatophore positions were exported* | import the dataset again: `python chromas_bridge.py import <chromas folder>\output\<dataset>.zarr` |
| *The lookup table labelled 9x% of the frame as chromatophore* | run CHROMAS again with the colour picker: the first colour is background, the rest are chromatophore (press **Clear all**, then **Suggest colours**) |
| *CHROMAS registration needs chromatophores to cover at least 5%* | run `extras\apply_patches.py` in `gillylab_chromas` (`setup.bat` does this; see its `INSTALL.md`) |
| *Not enough chromatophores detected (0 < 40)* | same as above, or try the `neuralnet` segmentation |
| *CHROMAS finished but wrote no dataset* | no in-focus stretch of 100 frames; lower the focus threshold |
| *no evidence that neighbours are more coupled* | a result, not an error. Check the traces first: if events are missed, adjust step 5 |
| Few chromatophores measured | read the `left_out` column of `chromatophores.csv` (outside the region, too small, area missing too often) |
| *Slicing failed! conflicting dimension sizes*, or *the array at index 1 has size …* | the chunks have different numbers of chromatophores: type the matching chunks (for example `0,2`) in the **Chunks** box of the Datasets tab and run the step again |
| *cluster_correlation could not converge after 10 attempts* (motor-unit analysis) | the dashboard retries with stronger damping, then with HDBSCAN; if it still fails, read the output box |
| *CHROMAS ran out of memory* | set `n_workers = 1` in `config.defaults.toml` |
| *Cannot synchronously wait on a running event loop* when CHROMAS finishes | harmless |

---

## What the method does, briefly

1. Each chromatophore's area is divided by its resting size, so small and large ones are judged alike.
2. An activation starts when the expansion rate rises above 4 times that chromatophore's own noise, and counts if it
   grows by at least 10 %. Times are in seconds.
3. Neighbours come from a triangulation of the chromatophore centres.
4. The coupling test compares neighbouring pairs with all others and repeats this 999 times with the chromatophores
   shuffled around the skin; the p value is the share of shuffles that do as well.
5. Activations on neighbours within 0.5 s are joined; groups larger than chance are waves, and a plane fitted to onset
   time against position gives their speed and direction.

The README section "Activation, neighbours and waves" explains each step.
