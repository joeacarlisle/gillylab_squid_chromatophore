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
| Hardware | the CHROMAS paper says training needs an NVIDIA GPU. It can start on a CPU after the patch below, but no CPU training run has been timed on this laptop; for real work use a GPU machine (for example a cloud GPU instance) |
| CHROMAS patch | run `gillylab_chromas\.venv\Scripts\python gillylab_chromas\extras\apply_patches.py` once. Without it, `chromas train` stops at once with the installed PyTorch (see "Known problems") |

How CHROMAS trains (read from its code, version 0.1.1): it reads pairs of PNG images and masks, cuts them into
512 x 512 windows overlapping by half, drops windows without any chromatophore, keeps 10 % of the windows for
validation, and trains an FCN ResNet-50 with random shifts, rotations, colour and brightness changes, perspective
distortion and flips. It saves the weights with the lowest validation loss. The binary cuttlefish model's own training
set in `Chromas2025.zip` is only 8 image/mask pairs, so a few dozen squid frames are a reasonable start.

---

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

**Using the trained network.** In the dashboard's CHROMAS tab choose *neuralnet* and put the new `.pth` file in the
*Weights file* box, or on the command line:

```
python chromas_bridge.py run <video> --method neuralnet --weights <chromas folder>\training_runs\squid_v1\fcn_resnet50_50_best_model.pth
```

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
| `-c` defaults to 4 classes | the supplied 2-class model does not load | always pass `-c 2` |
| Two preview windows open at the start and wait | the run cannot be left unattended until they are closed | close them; on a server without a screen this needs a code change |
| The output folder given with `-o` must exist | stops with an error | create it first |
| Validation windows come from the training frames | its validation scores are optimistic | score on separate test frames (step 5) |

The training patch was applied to the CHROMAS install on this computer on 13 November 2026. A full training run has
not been made yet on this laptop.

---

## The tools

| File | Environment | Does |
|---|---|---|
| `chromas_training.py frames / draft / masks / check` | this project | export frames, draft paint layers from a lookup table, convert paint layers to masks, check the pairs |
| `chromas_score_segmentation.py` | CHROMAS | score lookup tables and networks against hand masks |
| `chromas train` | CHROMAS | fine-tune or train the network |
| `gillylab_chromas\extras\apply_patches.py` | CHROMAS | the patches CHROMAS needs on this laptop, including the training patch |

`chromas_training.py` was tested on a squid skin video (frames, drafts, masks, check); its masks have the same format
as the masks in CHROMAS's own training set. `chromas_score_segmentation.py` was tested with the lookup method.