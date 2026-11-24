# Training the CHROMAS network on the lab's squid skin

CHROMAS can segment chromatophores with a neural network instead of a lookup table. The network that comes with it,
`sepia_2classes_fcn_resnet50.pth`, was trained on cuttlefish. This guide shows how to check it on the lab's own squid
videos and, if it falls short, fine-tune it on frames you paint by hand.

**Do this only if it is needed.** On isolated skin with dark chromatophores on a light background the
lookup method already works well. Training pays off when lookup fails: live animals,
uneven or changing light, chromatophores of several colours, or a lot of debris. Step 5 tells you whether it was worth it.

What you need:

| | |
|---|---|
| Time | about a day of painting for 20-30 frames (less when you correct drafts, step 2); training from an hour on a GPU to much longer on a CPU |
| Software | this project's environment (steps 1-3), the CHROMAS environment `gillylab_chromas\.venv` (steps 4-5), and an image editor with layers (GIMP is free) |
| Hardware | the CHROMAS paper says training needs an NVIDIA GPU. It can start on a CPU after the patch below, but no CPU training run has been timed on this laptop; for real work use a GPU machine (for example a cloud GPU instance). See "CPU or GPU: which PyTorch" below |
| CHROMAS patch | run `gillylab_chromas\.venv\Scripts\python gillylab_chromas\extras\apply_patches.py` once. Without it, `chromas train` stops at once with the installed PyTorch (see "Known problems") |

How CHROMAS trains (read from its code, version 0.1.1): it reads pairs of PNG images and masks, cuts them into
512 x 512 windows overlapping by half, drops windows without any chromatophore, keeps 10 % of the windows for
validation, and trains an FCN ResNet-50 with random shifts, rotations, colour and brightness changes, perspective
distortion and flips. It saves the weights with the lowest validation loss. The binary cuttlefish model's own training
set in `Chromas2025.zip` is only 8 image/mask pairs, so a few dozen squid frames are a reasonable start.

---

## Two ways to do it: the Training tab or the command line

Everything in this guide can be done with buttons in the dashboard's **6 Training** tab, or with the commands in
steps 1-5 below. They do the same things and write the same files, so you can mix them: export frames in the tab and
train on a GPU machine from the command line, for example. The command-line steps stay the reference, and they are what
you use on a computer without the dashboard.

### The Training tab

Start the dashboard (`scripts\windows\chromatophore_dashboard.bat`) and open **6 Training**. Work down the page; each
button runs one step, its output appears in the box at the bottom, and only one step runs at a time (**Stop** ends it).

| Tab step | What it does | Same as |
|---|---|---|
| **0. Before you start** | Says when training is worth it, and checks this computer: CHROMAS installed, the training patch applied, the supplied network found, an NVIDIA GPU or CPU only. **Check again** repeats it. | "What you need" |
| **1. Training and test sets** | Chooses the training set and the test set: folders under `training\`, written *group/name* (default `squid/train` and `squid/test`). **Create** makes a new one. It counts the frames, paint layers, drafts edited since they were made, and masks of both, and warns when the two sets are the same folder. The switch **Training set / Test set** below it says which set steps 2 to 5 work on. | step 1 |
| **2. Export frames** | Frames from a video, *spread over the video* (how many) or *at these seconds*, into `images\` of the set chosen by the switch. | `chromas_training.py frames` |
| **3. Drafts** | A lookup table's guess, as a red paint layer in `painted\`, for every frame that has none yet. It records when each draft was made, so step 1 can count the ones you have edited. | `chromas_training.py draft` |
| **4. Paint** | **Open the painted folder** (painting itself is done in GIMP or Fiji; *How to paint in GIMP* repeats step 2 below). **Painting rules** is a note saved with the set as `PAINTING_RULES.md`. | step 2 |
| **5. Masks and check** | Converts the paint layers to masks and checks the pairs; it ends with *Ready for chromas train*. Do it for both sets. | `chromas_training.py masks` and `check` |
| **6a. Random forest** | **Make crops**, then **Train random forest** (no GPU needed). | `chromas_training.py crops`, `chromas train -m rf` |
| **6b. Neural network** | Fine-tunes the network: **Start from** (the supplied network, a network you trained, or ImageNet only), **Passes** (`-e`), **Validate every** (`-v`), **Run name**. Below it, the same command to copy to a GPU machine, and **Results of a run** shows `metrics_plot.png` and `predictions.png`. | step 4 |
| **7. Score on the test set** | Scores any lookup table, random forests and networks on the test set and shows the mean Dice, IoU, precision, recall, chromatophores found and extra regions per method. The scores are also saved in the test set's folder as `scores_<date>_<time>.csv`. | step 5 |
| **8. Use the model** | Chooses *neuralnet* (a `.pth`) or *randomforest* (a `.pkl`) in the Run CHROMAS form of the **Datasets** tab and puts the model in its *Model file* box. | "Using the trained network" |

What the tab does for you, compared with the command line:

* step 6b always passes `-c 2` (two classes), creates the output folder `training_runs\<run name>\` before the run, and
  refuses a run name that is already used, so a run is never overwritten;
* it starts `chromas train` with Matplotlib set to draw off screen (`MPLBACKEND=Agg`), so the two preview windows should
  not open and wait (not yet confirmed on a real run on this laptop; see "Known problems");
* models go where `chromas train` puts them, `gillylab_chromas\training_runs\<name>\`; CHROMAS itself is not changed.

Keep the page open while a step runs. A network run that is stopped saves no network (CHROMAS writes it at the end).

### Example 1: fine-tune the network for the FASTEC recordings

1. **0. Before you start**: check that the training patch and the supplied network show a tick. "No NVIDIA GPU" means
   step 4 will run on the CPU: fine for a first try with few passes, slow for real work (see example 4).
2. **1. Sets**: leave `squid/train` and `squid/test`, or type `fastec/train` in *New set*, press **Create**, then
   `fastec/test` with the switch on *Test set*, and **Create** again.
3. **2. Export frames**, switch on *Training set*: video `data/FASTEC-IL5-714_2024-07-10_000029_h264.mp4`, *Spread over
   the video*, 8 frames, **Export frames**. Do the same for two or three other videos.
4. Switch to *Test set* and export 2 frames from a video **not** used for training, for example at chosen seconds:
   *At these seconds* `1.5 4`.
5. **3. Drafts**, for each set: lookup table `FASTEC-IL5-714_2024-07-10_000029_h264_colorchromas.lookup_table.npz` (one the
   colour picker saved), **Make drafts**.
6. **4. Paint**: write the rules, for example *"Every chromatophore, small ones too, as large as its pigment in this
   frame. Partly hidden: the visible part. Out of focus: leave out. The electrode, debris, bubbles: never."*, **Save the
   rules**. **Open the painted folder** and correct every draft in GIMP. Step 1 shows, for example, *8 paint layers (8 of
   8 drafts edited since they were made)* when all are done.
7. **5. Masks and check**, for the training set, then for the test set. Both should end with *Ready for chromas train*.
8. **6b**: *Start from* `supplied: sepia_2classes_fcn_resnet50.pth`, *Passes* 50, *Validate every* 5, *Run name*
   `fastec_v1`, **Train the network**. When it finishes, choose `fastec_v1` in *Results of a run* and look at
   `metrics_plot.png`: validation should improve and then level off.
9. **7. Score**: lookup table `FASTEC-..._colorchromas.lookup_table.npz`, networks `supplied: ...` and
   `trained: fastec_v1/fcn_resnet50_50_best_model.pth`, **Score on the test set**.
10. If the trained network wins (example 3), **8. Use the model**: choose `network: fastec_v1/...`, **Use it in Run
    CHROMAS**, then run CHROMAS on a video in the Datasets tab and check its overlay video.

### Example 2: a random forest instead (no GPU)

Steps 1 to 5 as in example 1, then **6a**: *Crop size* 512, *Crops per frame* 1, **Make crops** (into
`training\fastec\train_crops\`); *Train on* Crops, *Model name* `rf_fastec_v1`, **Train random forest** (more than ten
minutes on a small machine). Score it in step 7 under *Random forests*, next to lookup and the networks, and use it
with step 8 (it sets *randomforest*).

### Example 3: reading the scores

Step 7 shows one row per method, for example (made-up numbers, to show how to read it):

| method | Dice | IoU | precision | recall | found | extra |
|---|---|---|---|---|---|---|
| lookup (FASTEC-..._colorchromas.lookup_table.npz) | 0.71 | 0.55 | 0.63 | 0.82 | 118 of 124 | 41 |
| network (sepia_2classes_fcn_resnet50.pth) | 0.64 | 0.47 | 0.70 | 0.59 | 97 of 124 | 12 |
| network (fcn_resnet50_50_best_model.pth) | 0.83 | 0.71 | 0.85 | 0.81 | 120 of 124 | 9 |

Here lookup finds most chromatophores but marks a lot that is not one (low precision, 41 extra regions: probably the
electrode and debris); the supplied cuttlefish network misses many (low recall); the fine-tuned network has a clearly
higher Dice, far fewer extra regions and finds at least as many, so by the rule in step 5 it is worth using. If the
trained row were no better than lookup, keep lookup.

### Example 4: train on a GPU machine, use the result here

1. Do steps 1 to 5 in the tab, then copy the training set folder (for example `training\fastec\train\`) to the GPU
   machine, which needs the same CHROMAS install with the training patch and `model_weights\`.
2. Under **6b**, the line *On a GPU machine instead* shows the command for the settings you chose. On the GPU machine,
   change its paths to that machine's folders, create the output folder, and run it (the same command as step 4 below).
3. Copy the whole run folder back into `gillylab_chromas\training_runs\` on this laptop (it must hold
   `fcn_resnet50_<passes>_best_model.pth`). Press **Refresh lists**: it appears in *Results of a run*, *Start from*,
   step 7 and step 8 like a run made here.

---

## The command line

Steps 1 to 5 below are the same work done from a terminal.


## Step 1. Choose and export frames

Choose frames that cover what the network will meet:

* several animals or preparations, and several videos;
* chromatophores both relaxed and expanded (before, during and after a stimulus);
* the difficult parts: the electrode, debris, bubbles, iridescent patches, blur, bright and dark lighting;
* the same magnification as the videos you will analyse (CHROMAS wants at least about 10 pixels across a fully
  expanded chromatophore).

**Keep two sets apart from the start**: about 80 % of the frames for training and 20 % for testing, preferably from
different videos. The test frames are never shown to the network; they are what step 5 scores against. (CHROMAS's own
10 % validation comes from the same frames as the training, so it says little about new videos.)

Export the frames with this project's environment:

```
python chromas_training.py frames <your video> training\squid\train --times 0 3 4 10
python chromas_training.py frames <another video> training\squid\train --n 8
python chromas_training.py frames <a third video> training\squid\test --n 5
```

`--n` spreads the frames evenly over the video; `--times` takes them at the given seconds. Frames go to
`<folder>\images\` as `<video>_f<frame number>.png`. Frames must be at least 512 pixels on each side.

---

## Step 2. Paint the chromatophores

For each image you make a **paint layer**: chromatophores painted, everything else transparent. Save it under the
same file name in `<folder>\painted\`.

**Start from a draft (recommended).** Painting hundreds of chromatophores from nothing is slow; correcting a draft is
quick. A lookup table made for your video (the dashboard's colour picker saves one in the CHROMAS `output` folder as `<name>.lookup_table.npz`) gives drafts:

```
python chromas_training.py draft training\squid\train --lookup <your lookup_table.npz>
python chromas_training.py draft training\squid\test  --lookup <your lookup_table.npz>
```

This writes a red draft layer for each image into `painted\`. For other videos, make a lookup table for them first
(a CHROMAS lookup run saves `lookup_table.npz` next to its dataset), or paint from nothing.

**Painting in GIMP:**

1. Open the image (`images\<name>.png`).
2. *File > Open as Layers* and choose `painted\<name>.png` (the draft). Without a draft: *Layer > New Layer*, fill
   with *Transparency*.
3. Select the paint layer. Use the **Pencil** tool (N), not the brush: it has hard edges, so a pixel is either painted
   or not. Any colour; red is easiest to see. For removing paint use the **Eraser** with *Hard edge* ticked.
4. Zoom to 200-400 % and work across the image:
   * paint every chromatophore completely, as large as its pigment is in this frame, small ones too;
   * erase paint from the electrode, debris, bubbles, shadows and anything else that is not a chromatophore;
   * where two expanded chromatophores touch, paint both; CHROMAS separates them later by their territories.
5. Hide the image layer (eye icon) so only the paint layer is visible.
6. *File > Export As...*, `painted\<same name>.png`, overwrite the draft. (GIMP exports what is visible.)

Fiji / ImageJ works too: paint on a new image of the same size and save it as PNG under the same name.

Be consistent: the network learns exactly what you paint. Decide once whether partly hidden or out-of-focus
chromatophores count, write it down, and apply it everywhere. If more than one person paints, have both paint the
same two frames and compare (step 5's scores between the two show how consistent you are).

---

## Step 3. Convert and check

```
python chromas_training.py masks training\squid\train
python chromas_training.py check training\squid\train
python chromas_training.py masks training\squid\test
python chromas_training.py check training\squid\test
```

`masks` turns each paint layer into the mask format CHROMAS trains on: a single-channel PNG in `masks\` where
**0 = background and 1 = chromatophore**. These masks look completely black in an image viewer; that is correct.
`check` confirms that every image has a mask of the same size, at least 512 pixels on a side, with only 0 and 1, and
prints how much of each frame is chromatophore. It ends with *Ready for chromas train*.

(CHROMAS also has `chromas unify-training-data`, an interactive tool that turns colour-painted masks into this format.
`chromas_training.py masks` does the same for the paint-layer method above, without the clicking.)

---

## Step 4. Fine-tune the network

In the CHROMAS environment, starting from the supplied cuttlefish model:

```
cd <chromas folder>
.venv\Scripts\activate
mkdir training_runs\squid_v1
chromas train <squid folder>\training\squid\train ^
    -m fcn_resnet50 -c 2 --weights model_weights\sepia_2classes_fcn_resnet50.pth ^
    -e 50 -v 5 -o training_runs\squid_v1
```

| Option | Meaning |
|---|---|
| `-m fcn_resnet50` | the architecture of the supplied model |
| `-c 2` | **two classes, background and chromatophore. Required**: the default is 4, which does not fit the supplied model |
| `--weights ...` | start from the supplied model (fine-tuning). Leave it out to train from ImageNet weights only, which needs far more frames |
| `-e 50` | number of passes over the training windows (the CHROMAS default is 100) |
| `-v 5` | validate every 5 passes |
| `-o ...` | output folder; **it must exist already** |

At the start it shows two windows (one mask, one image); **close them** and training continues. The first run
downloads the ImageNet starting weights (about 135 MB) once. It prints the number of training and validation windows,
then the loss and scores as it goes.

The results in `training_runs\squid_v1\`:

* `fcn_resnet50_50_best_model.pth`: the trained network (the passes with the lowest validation loss);
* `metrics_plot.png`: loss, accuracy, IoU and Dice for training and validation over the passes. Validation should
  improve and then level off; if training keeps improving while validation gets worse, the network is memorising:
  use fewer passes or more frames;
* `predictions.png`: some validation windows with the hand mask and the prediction.

The learning rate is fixed in CHROMAS's code (0.001, lowered automatically when progress stalls). For fine-tuning a
lower rate is often better; if results are unstable, that is a change to make in its code.

---

## Step 5. Score lookup against the network on the test frames

In the CHROMAS environment, with the test folder from step 1:

```
.venv\Scripts\python <squid folder>\chromas_score_segmentation.py ^
    <squid folder>\training\squid\test ^
    --lookup <your lookup_table.npz> ^
    --weights model_weights\sepia_2classes_fcn_resnet50.pth training_runs\squid_v1\fcn_resnet50_50_best_model.pth ^
    --out training_runs\squid_v1\test_scores.csv
```

It segments every test frame with each method and compares it with your painted mask:

| Score | Meaning |
|---|---|
| IoU, Dice | pixel overlap with your mask, 0 to 1 (1 = identical). Dice is the more forgiving of the two |
| precision | share of what the method calls chromatophore that really is (low = it marks debris, electrode, background) |
| recall | share of the real chromatophore pixels it finds (low = it misses or shrinks chromatophores) |
| found | chromatophores in your mask that the method touches at all |
| extra | regions the method marks that are not chromatophores |

**How to decide.** Use the fine-tuned network if, on the test frames, it has a clearly higher Dice than lookup, fewer
*extra* regions, and finds at least as many chromatophores. If lookup is as good, keep lookup: it is faster and needs
no GPU. Then check the effect on the actual question: run CHROMAS on a test video with each segmentation and compare
`chromas_activation.py` results (events, coupling, waves). Conclusions that change with the segmentation are not yet
solid.

**Using the trained network.** In the dashboard's **Datasets** tab (Run CHROMAS form) choose *neuralnet* and put the new `.pth` file in the
*Model file* box (step 8 of the **Training** tab does both for you), or on the command line:

```
python chromas_bridge.py run <video> --method neuralnet --weights <chromas folder>\training_runs\squid_v1\fcn_resnet50_50_best_model.pth
```

---

## CPU or GPU: which PyTorch

`setup.bat` installs the **CPU-only build** of PyTorch in the CHROMAS environment (`2.15.0+cpu`), because it is a much
smaller download. That build can never use a GPU, even on a computer that has one; step 0 of the Training tab says so,
and *Use GPU* in the Datasets tab is greyed out.

**On a computer without an NVIDIA graphics card, keep the CPU build.** A CUDA build would also run on the CPU there, so
it gives nothing extra, after a download of several gigabytes. Train the random forest here, try the network with few
passes, and do real network training on a GPU machine (Training tab, example 4).

**On a computer with an NVIDIA card**, a CUDA build of PyTorch gives GPU and CPU support in one install: it uses the GPU
when one is there and the CPU when it is not, because CHROMAS's training, scoring and (patched) neural-net segmentation
all choose "GPU if available, otherwise CPU". Only one card is used: the first one (set `CUDA_VISIBLE_DEVICES` before
starting the dashboard or a command to choose another). Only PyTorch is replaced; CHROMAS and its other packages stay.

PyTorch 2.15.0 for Windows and Python 3.12 is published as CUDA 13.2 (`cu132`) and CUDA 13.4 (`cu134`) builds. The
installed `torchvision 0.29.2` and `torchaudio 2.11.1` work with any PyTorch 2.14 or newer, so they stay.

1. **Check the NVIDIA driver.** Run `nvidia-smi`; *CUDA Version* in its top right corner must be at least 13.2 for the
   `cu132` build (13.4 for `cu134`). If it is lower, update the driver first.
2. **Replace PyTorch** in the CHROMAS environment (a download of several gigabytes):

   ```
   cd C:\Users\joeac\gillylab_chromas
   .venv\Scripts\activate
   pip install torch==2.15.0+cu132 --index-url https://download.pytorch.org/whl/cu132 --force-reinstall --no-deps
   ```

   For CUDA 13.4 write `cu134` in both places. `--no-deps` replaces only PyTorch and leaves the other packages as they are.
3. **Check it:**

   ```
   .venv\Scripts\python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

   `2.15.0+cu132 13.2 True` means the GPU is ready; `False` means no usable card or driver (it then runs on the CPU).
4. **Restart the dashboard.** Step 0 of the Training tab shows the card with a tick, and *Use GPU* in the Datasets tab
   can be ticked again.

**To go back to the CPU build:**

```
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

**Note:** running `setup.bat` again puts the CPU build back (its step 4 installs from the CPU index). After rebuilding
the environment on a GPU machine, repeat step 2.

These commands were not run on this laptop (it has the CPU build); the build names come from PyTorch's package index.

---

## Keep the training data

Store `training\squid\` (images, painted layers, masks, and a note of the painting rules) with the lab's data, for
example in the data repository. The masks are small. With them, every later model, or a newer CHROMAS, can be scored
on exactly the same frames, and the network can be retrained when new kinds of video come in.

---

## Known problems in CHROMAS 0.1.1 training

| Problem | Effect | What to do |
|---|---|---|
| The learning-rate scheduler is created with `verbose=True`, which the installed PyTorch (2.15) no longer accepts | `chromas train` stops at once with a TypeError | fixed by `extras\apply_patches.py` (training patch) |
| Starting weights saved on a GPU are loaded without `map_location` | fails on a computer without a GPU | fixed by the same patch |
| Masks must hold class numbers (0, 1) in the first channel; the colour table in its code is never used | colour-painted masks give nonsense classes | `chromas_training.py masks` (or `chromas unify-training-data`) |
| `-c` defaults to 4 classes | the supplied 2-class model does not load | always pass `-c 2` (the Training tab does) |
| Two preview windows open at the start and wait | the run cannot be left unattended until they are closed | close them; on a server without a screen, set `MPLBACKEND=Agg` before `chromas train` (the Training tab does this; not yet confirmed on a real run) |
| The output folder given with `-o` must exist | stops with an error | create it first (the Training tab does) |
| Validation windows come from the training frames | its validation scores are optimistic | score on separate test frames (step 5) |

The training patch was applied to the CHROMAS install on this computer on 13 November 2026. A full training run has
not been made yet on this laptop.

---

## The tools

| File | Environment | Does |
|---|---|---|
| `training_view.py` (the **6 Training** tab) | this project | all the steps of this guide with buttons; it runs the tools below |
| `chromas_training.py frames / draft / masks / check` | this project | export frames, draft paint layers from a lookup table, convert paint layers to masks, check the pairs |
| `chromas_score_segmentation.py` | CHROMAS | score lookup tables and networks against hand masks |
| `chromas train` | CHROMAS | fine-tune or train the network |
| `gillylab_chromas\extras\apply_patches.py` | CHROMAS | the patches CHROMAS needs on this laptop, including the training patch |

`chromas_training.py` was tested on a squid skin video (frames, drafts, masks, check); its masks have the same format
as the masks in CHROMAS's own training set. `chromas_score_segmentation.py` was tested with the lookup method. The
Training tab was tested with its steps 1 to 5 on a real video and with steps 6 to 8 against a stand-in for CHROMAS; a
real network training run from the tab has not been made yet.