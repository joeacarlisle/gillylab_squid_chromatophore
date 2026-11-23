# Gilly Lab Squid Chromatophores

An interactive dashboard for analysing chromatophores in close-up skin video of squid and other cephalopods with [CHROMAS](https://elifesciences.org/articles/106509) (Ukrow et al. 2025). From one browser window you prepare a video, run CHROMAS (segmentation, tracking, area of every chromatophore over time), and explore the result: chromatophore maps and traces, activation and waves, wedge (slice) traces, motor units and groups of chromatophores that move together.

A separate **Signal** add-on measures the brightness of boxes on the whole animal in distant footage (flashing and flickering in free-swimming Humboldt squid, *Dosidicus gigas*). It does not use CHROMAS and is described in Part 2 below.

**Demo data.** The Signal add-on comes with a synthetic demo video (two drawn shapes with known flashing and flickering) and four short real clips (Movies 1 to 4 of Rosen et al. 2015; see *Real clips used for testing* and the *Acknowledgment*).

## Which part do you need?

| | **CHROMAS analysis** (the main part) | **Signal add-on** |
|---|---|---|
| For | close-up skin video, one chromatophore at a time | distant footage of a whole animal |
| Measures | area of every chromatophore over time, activation, waves, motor units, groups | brightness of boxes on the animal: flashing, flickering, phase |
| Needs | the separate CHROMAS install (`gillylab_chromas`) | nothing besides this project |
| Dashboard tabs | Video, Datasets, Chromatophores, Activation, Advanced analytics | Signal |
| Database | `processed/gl_chromas.duckdb` | `processed/gl_chromatophore.duckdb` |
| Demo | none | `chromatophore_signal_demo` script (see *Signal demo*) |

The installation, the start scripts and the dashboard are shared; everything else is in one of the two parts below.

## Installation

### Root Directory Notes

The project root can be named anything (e.g., `gillylab_squid_chromatophore`).

To avoid Windows permission issues, place the root folder inside your Windows user directory:
`C:\Users\<your_username>\gillylab_squid_chromatophore`

### Python Installation Requirements

This project requires **Python 3.13**.

#### Windows

Install Python 3.13 from python.org (not from the Microsoft Store, which puts it in a different folder than the instructions below assume).

1. If you installed Python 3.13 from the Microsoft Store, remove it first: Windows Settings, Apps, "Python 3.13", Uninstall.
2. Download the "Windows installer (64-bit)" for Python 3.13 from https://www.python.org/downloads/windows/.
3. Run it and choose **Install Now**. Leave **"Install for all users"** unticked, so it installs for your user only. Leave the **py launcher** ticked.
4. When the installer asks to **"Add python.exe to PATH"**, select **No** (leave it unticked).
5. Python installs to `C:\Users\<your_username>\AppData\Local\Programs\Python\Python313\`.
6. Check it in Command Prompt: `py -0p` should list `Python313` under `AppData\Local\Programs\Python`.

#### macOS (Homebrew)

    brew install python@3.13

#### Linux (Ubuntu / Debian) using apt install

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.13 python3.13-venv

#### Linux (Ubuntu / Debian) using pyenv

This has the advantage of not requiring root privileges, and it does not install Python 3.13 system wide.

    # Install pyenv:
    curl -fsSL https://pyenv.run | bash
    # Install Python 3.13:
    pyenv install 3.13

### Prerequisites

**Git** must be installed to download the software from GitHub (`git clone`). Download it from [git-scm.com](https://git-scm.com/downloads) and confirm with `git --version`.

Python 3.13 is required (see above). All dependencies, including OpenCV and a bundled ffmpeg, are declared in `pyproject.toml` and installed from PyPI when you run `pip install -e .` - no manual download required.

### Getting Started

    git clone https://github.com/joeacarlisle/gillylab_squid_chromatophore.git
    cd gillylab_squid_chromatophore

Then follow the **Prerequisites** and **Setup** sections to install Python 3.13 and the project dependencies.

### Setup: Virtual Environment and Installation

*Run from the project root directory.*

#### Windows (Command Prompt)

The quick way (needs Python 3.13; `winget install Python.Python.3.13`):

    cd C:\Users\<your_username>\gillylab_squid_chromatophore
    setup.bat

`setup.bat` creates `.venv` with Python 3.13 and runs `pip install -e .`. `scripts\windows\setup.bat` does the same and can be started from any folder. To do it by hand:

    cd C:\Users\<your_username>\gillylab_squid_chromatophore
    C:\Users\<your_username>\AppData\Local\Programs\Python\Python313\python.exe -m venv .venv
    .venv\Scripts\activate
    pip install -e .

#### macOS / Linux

The quick way (needs `python3.13` on the PATH):

    cd gillylab_squid_chromatophore
    ./setup.sh

`setup.sh` creates `.venv` and runs `pip install -e .`. `scripts/linux_mac/setup.sh` does the same and can be started from any folder. To do it by hand:

    cd gillylab_squid_chromatophore
    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install -e .

### Running the dashboard

* **Windows:** `scripts\windows\chromatophore_dashboard.bat`
* **Linux / macOS:** `./scripts/linux_mac/chromatophore_dashboard.sh`

Or run `python -m panel serve chromatophore_holoviews.py --port 5009 --static-dirs videos=videos --show` from the project root with the virtual environment activated. The `--static-dirs` option lets the browser play the videos. The dashboard runs on port 5009 (http://localhost:5009/chromatophore_holoviews).

The tabs come in two groups, each following its own choice in the sidebar: the CHROMAS tabs (Part 1) and the Signal tab (Part 2).

## Part 1 - CHROMAS analysis (single chromatophores)

[CHROMAS](https://elifesciences.org/articles/106509) (Ukrow, Renard, Moghimi and Laurent, eLife 2025) segments and tracks individual chromatophores in close-up skin video and measures the area of each over time. This part of the dashboard runs CHROMAS for you and shows what comes out, so everything is done from the dashboard. CHROMAS needs about 10 pixels across a fully expanded chromatophore, so it suits close-up skin video such as the authors' example videos, not distant footage of a whole animal (for that, see Part 2). The CHROMAS paper is in `gillylab_chromas/docs/chromas-v2.pdf`.

**Quick start**

1. Install CHROMAS from `gillylab_chromas` (see its `INSTALL.md`) next to this folder, and this project (see *Installation*).
2. Start the dashboard (see *Running the dashboard*). It opens on the **Datasets** tab.
3. In the **Video** tab, check a video and prepare it if needed; in the **Datasets** tab, run CHROMAS on it. The result is chosen in the sidebar by itself.
4. Explore it in the **Chromatophores**, **Activation** and **Advanced analytics** tabs.

### What it does

* **Runs CHROMAS from the dashboard:** the dashboard starts CHROMAS in its own environment (chunking, segmentation, registration, stitching, area), then loads the result and chooses it in the sidebar.
* **Three segmentation methods:** a **lookup table** (colours you choose in the built-in colour picker; no weights needed), a **random forest** trained on your own painted frames (a form in the Datasets tab takes you through it step by step), and the CHROMAS **neural network** (needs the trained weights, which are not shipped).
* **Stores the results in DuckDB:** the area of every chromatophore in every frame, the focus check and a record of each run go to `processed/gl_chromas.duckdb` (tables `chromas_runs`, `chromas_areas` and `chromas_focus`). Everything else the tabs use (positions, quality checks, wedges, motor units, activation results) stays in files in `processed/chromas_export/<dataset>/` and `processed/chromas_activation/<dataset>/`. DuckDB is pinned to 2.0, because a file written by 2.0 cannot be opened by 1.x.

### The CHROMAS tabs

The sidebar shows the chosen **CHROMAS dataset** and how far it is (in the database, positions, analysed), with what to do next.

| Tab | What it is for |
|---|---|
| **Video** | Check codec, frame rate and exact frame count, place and check a crop, and cut or convert the video for CHROMAS with ffmpeg. The original is never changed. |
| **Datasets** | Run CHROMAS on a video, import a `.zarr` result or results exported earlier, and run the extra steps a dataset needs: quality checks, the slice step, CHROMAS grouping, the motor-unit analysis, quality-control videos, and **Measure areas again**. A table lists every dataset with its next step and how good the run is: chromatophores measured out of numbered (in total and per chunk), median area per chunk, whether the chunks agree, frames in focus, the lookup table used, and which later steps are done (columns show yes or no; stale means a step was run before the areas were measured again). |
| **Chromatophores** | A map of the chromatophores coloured by a measure (click one, or drag a box, to plot it); the area of each over time and a heat map; histograms of resting size, largest expansion and time expanded; groups that move together; flower plots (after the slice step); and quality checks: focus over time, pigment coverage, registration between chunks, sample frames next to their segmentation, and overlay videos. |
| **Activation** | Activation events per chromatophore, whether neighbours are activated together, wave speed and direction, and a replay next to the video (see below). |
| **Advanced analytics** | Reports modelled on the figures of the CHROMAS papers: the 36 wedge traces of a chromatophore, motor units (the papers' method), wedge correlation clusters (a shortcut of this project, not motor units), cluster statistics and network, wedge orientation, CHROMAS's own groups, expansion against relaxation speed, and the overlap of two chunks. The wedge reports need the slice step. |

### Choosing the colours (lookup method)

For a step-by-step guide to getting the best results with this method, with how to read the dataset table and improve a run, see [docs/TUTORIAL_LOOKUP.md](docs/TUTORIAL_LOOKUP.md).

* **Colour picker (recommended).** Load a frame and click the colours yourself (**Background** first, then **Chromatophore**), or press **Suggest colours** to replace your clicks with colours found in the frame. Check the preview, correct it with your own clicks (any number of **Background** and **Chromatophore** clicks), step through other suggestions with **Next suggestion** and **Previous suggestion**, then press **Use these colours and run CHROMAS**.
* **CHROMAS's own window.** Press **Run CHROMAS** with the lookup method and CHROMAS opens its window on the first frame. Click the background first, then at least one chromatophore, then press Q. Only the first click is background.
* **Running the same video again.** A run with the same dataset name replaces that dataset. The colours are saved for the video and **Run CHROMAS** reuses them without opening a window; choose new colours in the colour picker below. A copy of the colours behind each result is kept in `processed/chromas_export/<dataset>/lookup_table.npz`.
* **Measure areas again** (Datasets tab) repeats CHROMAS's area step with another minimum territory area, border and mask widening (defaults 200 px, 40 px and 10) without a new run. CHROMAS keeps only the largest connected group of pigmented regions as the body and gives territories only inside it, so chromatophores far from that group (for example separate patches in a close-up) get none; a larger **mask widening** (try 30 to 40) joins them to it, and also adds territory in the gaps. Territory numbers can change, so run the slice, grouping and motor-unit steps again afterwards.
* **Focus threshold and Min chunk** (Run CHROMAS form). CHROMAS drops frames below the focus threshold (default 0.01) and keeps only in-focus stretches of at least **Min chunk** frames (default 100). A short or soft video can have no stretch that long: the run then stops with an error, and a lower threshold or a lower Min chunk (for example 50) fixes it.
* **Run bar and Stop.** While CHROMAS runs, a bar with the elapsed time and the step's percentage appears under **Run CHROMAS**, with a **Stop** button that ends the process. Output printed so far stays in the output box, and a partly written result can be replaced by running the video again with the same dataset name.

### CHROMAS demos

* **Check the activation method on synthetic data** - the **Check the method** button in the Activation tab builds two synthetic patches with known waves and checks that the analysis finds them.
* **Check or fine-tune the neural network on your own squid video** - [docs/TRAINING_THE_NETWORK.md](docs/TRAINING_THE_NETWORK.md) (not a demo; do it only if the lookup method falls short).

### How it fits together

CHROMAS lives in its own project, `gillylab_chromas` (Python 3.12, its own `.venv`), because it needs `opencv-python` 4.x and this project pins `opencv-python-headless` 5.0: both write to the same `cv2` folder. The dashboard never imports CHROMAS; it starts it as a separate process:

    video  --chromas run-->  output/<name>.zarr  -->  exported files  -->  processed/gl_chromas.duckdb  -->  dashboard

The CHROMAS results have their own database file because the dashboard holds `gl_chromatophore.duckdb` (Signal) open read-only, and DuckDB does not allow another process to write to a file that is open. The location of the CHROMAS project is `[chromas] home` in `config.defaults.toml` (default `../gillylab_chromas`, next to this folder; the environment variable `GL_CHROMAS_HOME` overrides it).

### Activation, neighbours and waves

The **Activation** tab turns the areas and positions that CHROMAS measures into activation events, neighbour coupling tests and waves. Choose a dataset in the sidebar and press **Analyse dataset**. If the dataset has its own `examples/<name>/` folder with an `example.toml`, its settings (scale and region) are chosen by itself; otherwise choose a settings file and adjust the thresholds. The tab shows the result in words, the coupling test and coupling against distance, every activation onset (a wave is a slanted line of dots), the area traces with their onsets, the waves with an activation map of each, and a **replay**: the chromatophores animated over time next to the same moment of the video, with a ring at each activation onset coloured by its wave. The analysis builds on earlier analysis code from the Gilly lab; thanks to its authors.

**The method**

1. *Relative size.* Each chromatophore's area is divided by its resting size (10th percentile) minus 1, so 0.3 means 30 % larger than at rest. One threshold then suits small and large chromatophores.
2. *Activation events.* The expansion rate must rise above 4 times that chromatophore's own noise, and the chromatophore must grow by at least 10 % of its resting size. The onset is where the rate first left the noise, with sub-frame precision. Times are in seconds, so the settings do not depend on the frame rate.
3. *Neighbours.* A Delaunay triangulation links each chromatophore to the ones around it, whatever the density; long edges are dropped.
4. *Are neighbours coupled?* Two measures per pair: the correlation of the expansion rates, and the share of events that coincide within 0.2 s. The neighbour mean minus the mean of all other pairs is compared with 999 shuffles of the signals among the positions, which gives a p value.
5. *Waves.* Events on neighbouring chromatophores within 0.5 s of each other are joined into groups. A group counts as a wave only if it is larger than what the same events form by chance. A plane fitted to onset time against position gives the speed and direction; the wave is called travelling if R^2 >= 0.5 and a shuffle test gives p < 0.05.

Two further checks are included: a whole-patch synchrony test (a whole-body display such as Humboldt squid flashing gives no neighbour excess, so the neighbour test alone would look empty), and a cross-check event detector that picks peaks by prominence, after Renard, Ukrow et al. 2026.

**Results** are written to `processed/chromas_activation/<dataset>/`: `summary.txt` (the result in plain language), tables of events, coupling, neighbour pairs, waves and areas, and pictures of the pathways and of each wave.

**Limits.** Each CHROMAS chunk is analysed on its own (chromatophore numbers can differ between chunks). Positions are in CHROMAS's master frame, which is assumed to have the video's pixel grid for the region and the pixels per mm. The thresholds are starting values; check them on clips the lab has scored by hand.

### Network weights

The network weights (`sepia_2classes_fcn_resnet50.pth`, 141 MB) are not part of this repository. They come from the CHROMAS authors' data release (Ukrow et al. 2025; Edmond, doi:10.17617/3.MJI3HA); check the licence there before you share the file. Put the file in `models/`; if `models/` has none, the dashboard uses `model_weights/` in the CHROMAS folder.

The provided models were trained on manually annotated images of bobtail squid (*Euprymna berryi*) and cuttlefish (*Sepia officinalis*), not on Humboldt squid, so check them on your footage before relying on them. The paper says the neural-network method needs an NVIDIA GPU; this install runs it on the CPU through CHROMAS patch 2 (see `INSTALL.md` in the CHROMAS folder), which is slow. If you add large files of your own, `.gitattributes` stores `.pth` and `.pt` files with [Git LFS](https://git-lfs.com); run `git lfs install` once per computer.

## Part 2 - Signal add-on (box brightness on the whole animal)

A separate add-on (the **Signal** tab) for distant footage of a whole animal, such as the Humboldt squid clips of Rosen et al. (2015). It does not use CHROMAS and has its own database. The analysis is simple: it follows the brightness of one or more boxes on the animal and labels each moment as quiet, flickering or flashing.

### What it does

* **Brightness per box:** for each ROI box (a patch of skin) the mean grayscale value is taken for every video frame, the quantity Rosen et al. (2015, J Exp Biol 218:265) measured by hand.
* **States:** a sliding window (2 s) is analysed with an FFT. Each window is labelled **quiet**, **flickering** (broadband activity with no single rhythm) or **flashing** (one sharp rhythm, 1.5 to 5 Hz by default).
* **Measures:** dominant frequency and swing for every window, episodes (start, end, duration, mean flash frequency), and the phase between two animals when both flash. Cycle-by-cycle frequency, pauses and spatial synchrony are also written to the database.
* **Optional tracking:** a box can follow the animal through the video (optical flow, translation only). Windows where tracking was lost are labelled **untracked**.
* **DuckDB storage:** results go to `processed/gl_chromatophore.duckdb`, with the settings used and the video's SHA-256, separate from the CHROMAS database.

### Using it

1. Put an H.264 `.mp4` in `videos/` (browsers play only H.264). To convert another format: `ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p videos/<video_id>.mp4`. The video id is the file name without `.mp4`.
2. Open the **Signal** tab, choose the video in **Analyse a video in gillylab_squid_chromatophore\videos\\** (or type the full path of a file elsewhere) and press **Add and analyse**. The video plays at the top, with the signal and detected states, spectrogram, episodes, phase and tracking below it, and a Setup section that shows the ROI boxes on the first frame.
3. Without a `.toml` file the analysis uses one box over the whole frame. To measure particular patches, edit `videos/<video_id>.toml` and analyse the video again. It has one box per animal, as `name = [x, y, width, height]` in pixels from the top-left corner of the frame:

        [rois]
        squid_1 = [320, 210, 120, 60]
        squid_2 = [610, 180, 100, 50]

   The files in `videos/` show the optional `[tracking]` and `[grids]` tables. An open-water box (for example `background`) is a useful reference: if it oscillates like the skin, the rhythm comes from the camera or the light, not from the animal.

### Signal demo

`scripts\windows\chromatophore_signal_demo.bat` (Linux / macOS: `./scripts/linux_mac/chromatophore_signal_demo.sh`) regenerates the synthetic two-squid demo video (about 30 seconds) and analyses it. The video has known quiet, flickering and flashing periods; start the dashboard and tick **Show known truth** in the Signal tab to compare them with what was detected. The first time the dashboard starts, it analyses the videos in `videos/` by itself.

### Settings

The window length and the thresholds are in `config.defaults.toml`; a video's own `.toml` can override them. They are placeholders: they were tuned on the synthetic demo and must be checked against hand-labelled footage before the numbers are used for anything scientific.

### Real clips used for testing

`videos/jeb114157_video1` to `jeb114157_video4` are Movies 1 to 4 of Rosen et al. (2015), grayscale Crittercam footage at 15 fps, included for testing only (see the *Acknowledgment*). With placeholder thresholds and fixed boxes they give plausible output, not a quantitative match to the paper. In Movie 1 the mantle box reads as flashing from about 3.4 s, but the background box shows a similar peak near 2.7 Hz, which points to camera shake or lighting, so check that clip by eye before treating it as flashing.

### Limits

* The thresholds are placeholders tuned on the synthetic demo, so the demo is not an independent test.
* Boxes are fixed unless tracking is switched on. Tracking follows translation only and drifts when the animal turns or leaves the frame.
* Only grayscale brightness is used; colour and lighting changes are not separated from brightness. Camera movement and swimming change every box.
* One frequency per window, and state changes are resolved only to about one window (2 s).
* The real clips are 4 to 15 s long, too short for statistics.
* The dashboard does not yet draw the cycle, pause and grid results; they are in the database.

## Project structure

*`processed/` and `logs/` are created automatically on first run.*

```
.
|   setup.bat, setup.sh                 creates .venv and installs the project (Windows; Linux / macOS)
|   pyproject.toml
|   config.defaults.toml                settings: the [chromas] table, and the Signal thresholds
|   chromatophore_holoviews.py          the dashboard (every tab)
|   plot_tools.py                       a full-screen button on every dashboard plot
|   dataset_state.py                    the CHROMAS dataset chosen in the sidebar, shared by the CHROMAS tabs
|
|   --- CHROMAS analysis ---
|   chromas_bridge.py                   runs CHROMAS in its own environment and loads its results into DuckDB
|   chromas_export.py                   reads CHROMAS Zarr output (runs in the CHROMAS environment)
|   import_view.py                      the Datasets tab (run CHROMAS, import its results)
|   rf_training_view.py                 the training form of the Datasets tab (train a random forest step by step)
|   video_view.py, video_prep.py        the Video tab and the ffmpeg / ffprobe steps behind it
|   chromas_view.py                     the Chromatophores tab and the Run CHROMAS form
|   chromatophore_views.py              the Chromatophores tab's map, distributions, groups, flowers and quality checks
|   chromas_activation.py               activation events, neighbour coupling and waves on CHROMAS results
|   activation_view.py, replay_view.py  the Activation tab and its replay
|   supplemental_view.py                the Advanced analytics tab (reports modelled on the CHROMAS paper figures)
|   chromas_training.py                 export frames, draft and convert hand-painted masks for training
|   chromas_score_segmentation.py       score lookup / random forest / network against hand masks (CHROMAS environment)
|   make_fake_chromas.py                writes a synthetic CHROMAS-style dataset for trying the bridge
|   docs/TUTORIAL_LOOKUP.md              tutorial: getting the best results with the lookup method
|   docs/TRAINING_THE_NETWORK.md        painting training frames, fine-tuning and scoring the CHROMAS network
|   models/                             put the CHROMAS network weights here (not in git)
|
|   --- Signal add-on ---
|   main.py                             orchestrator: video + ROIs -> DuckDB
|   chroma_analysis.py                  windowed FFT, states, phase, episodes, flash cycles, pauses, grid synchrony (numpy, pandas, scipy)
|   roi_tracker.py                      ROI box that follows the animal (OpenCV optical flow)
|   make_demo_video.py                  synthetic two-squid demo video with known truth
+---videos
|       demo_two_squid.mp4              synthetic demo video (small, kept in git)
|       demo_two_squid.toml             its ROI boxes
|       demo_two_squid_truth.json       what the generator drew (shown in the dashboard)
|       jeb114157_video1..4.mp4/.toml   real test clips (Rosen et al. 2015) and their ROIs
+---processed                           created automatically on first run
|       gl_chromas.duckdb               CHROMAS results
|       chromas_export/<dataset>/       CSV files exported from CHROMAS (areas, focus, centroids), outline picture, info.json
|       chromas_activation/<dataset>/   results of chromas_activation.py
|       gl_chromatophore.duckdb         Signal results (separate file)
|       <video_id>_roi_preview.png      Signal: first frame with the ROI boxes drawn on it
+---logs                                created automatically on first run
\---scripts
    +---linux_mac   chromatophore_signal_demo.sh, chromatophore_dashboard.sh, setup.sh
    \---windows     chromatophore_signal_demo.bat, chromatophore_dashboard.bat, setup.bat
```

### File descriptions

**Shared**

* **chromatophore_holoviews.py** - Read-only Panel dashboard.
* **config.defaults.toml** - Analysis settings. The Signal thresholds are described in Part 2, *Settings*.

**CHROMAS analysis**

* **chromas_bridge.py** - Starts CHROMAS (installed in `gillylab_chromas`) as a separate process, starts the export step, and loads the results into `processed/gl_chromas.duckdb`.
* **chromas_export.py** - Reads a CHROMAS Zarr dataset and writes CSV, PNG and JSON files. It runs in the CHROMAS environment, not this one.
* **chromas_view.py** - The form that starts a CHROMAS run (in the Datasets tab), and the Chromatophores tab: plots of the chromatophore areas.
* **video_view.py**, **video_prep.py** - The Video tab (the first tab). **Check video** (codec, size, frame rate, exact frame count), **Show full frame / Show frame with crop / Show 4 frames / Check crop** (place the crop; the **Preview time** slider chooses the frame), **Make H.264 copy and check it** (an H.264 `.mp4`, optionally cropped or trimmed, written to the CHROMAS data folder and checked against the original; a record with the checksum of the original goes to `processed/video_prep/`), and **Make clicking copy at the preview time** (a copy whose first frame is the frame at the preview time, optionally reversed, for clicking lookup colours on large chromatophores; use it only for clicking). The original is never changed.
* **chromas_activation.py** - Finds activation events in the CHROMAS areas, decides whether neighbouring chromatophores are activated together (permutation test), and measures the speed and direction of waves. See *Activation, neighbours and waves*.
* **make_fake_chromas.py** - Writes a synthetic CHROMAS-style dataset (60 chromatophores, 2 chunks) so the bridge can be tried before there is real CHROMAS output.

**Signal add-on**

* **main.py** - Orchestrator. Reads each video with OpenCV, extracts the mean grayscale per ROI and frame, runs the analysis, and writes the DuckDB atomically with build provenance.
* **chroma_analysis.py** - The analysis itself (no video or database code): window features, state classification, phase between ROIs and the in-phase share, episodes, flash cycles (frequency and amplitude), pauses, and grid synchrony.
* **roi_tracker.py** - `BoxTracker`: follows a box with the median motion of feature points inside it (Lucas-Kanade optical flow, each point checked by tracking it backwards).
* **make_demo_video.py** - Draws the synthetic demo video and writes the truth file. Uses the ffmpeg bundled with `imageio-ffmpeg` to write H.264, which plays in every browser.
* **videos/&lt;video_id&gt;.toml** - Per-video ROI boxes (written automatically as one whole-frame box if missing), optional `[tracking]` and `[grids]` tables, and optional overrides of any setting.

## References and acknowledgments

### CHROMAS

Ukrow J, Renard N, Moghimi M, Laurent G (2025). eLife 14:RP106509. https://elifesciences.org/articles/106509

The motor-unit analysis follows Renard et al. (2026), eLife RP110074. CHROMAS is released under the 3-Clause BSD licence; `gillylab_chromas` adds an installer and two small patches (see its README) and is not by the CHROMAS authors.

### Signal add-on

Rosen H, Gilly W, Bell L, Abernathy K, Marshall G (2015). Chromogenic behaviors of the Humboldt squid (*Dosidicus gigas*) studied in situ with an animal-borne video package. Journal of Experimental Biology 218: 265-275. https://doi.org/10.1242/jeb.114157

#### Acknowledgment

This project is built on the work of Rosen, Gilly, Bell, Abernathy and Marshall (2015), cited above. The idea of measuring skin brightness in a box on the animal, the definitions of flashing and flickering, the 2 to 4 Hz flash range, the cycle-by-cycle frequency, the pauses, and the grid of small boxes for spatial synchrony all come from their paper. The four real clips used for testing (`jeb114157_video1` to `jeb114157_video4`) are the paper's supplementary Movies 1 to 4, recorded with National Geographic's Crittercam on Humboldt squid in the Gulf of California. The footage belongs to the authors and the publisher and is included here only to test the method; the rights stay with them. This is an independent reimplementation; any errors in it are not theirs.

## Dashboard stops responding

The dashboard stops responding after a period of inactivity and needs to be refreshed (press F5 or reload the page). This is the browser's tab-sleeping setting, on Windows and on Mac: Edge puts an inactive tab to sleep (15 minutes on the Windows machine this was tested on; the time is configurable), and Chrome's Memory Saver can do the same. Either one cuts the tab's link to the dashboard server. In Edge, add the dashboard address to Settings > System and performance > Performance > "Always keep these sites active" (for example `localhost`), or turn off "Automatically put tabs to sleep". In Chrome, use Settings > Performance > "Always keep these sites active", or turn off Memory Saver.

## Contact

Joe Carlisle - joeacarlisle@gmail.com

