# crane-annotation-demo

**A small, self-contained demo of a bioacoustic annotation workflow, run on an openly licensed Sandhill Crane recording.**

This notebook walks the basic path every bioacoustics annotation project shares: load a field recording, look at it as a spectrogram, detect call onsets and offsets, and export the events as a selection table that opens directly in [Raven / Raven Lite](https://www.ravensoundsoftware.com/).

It exists as a public companion to [GRUS](https://github.com/RobotWhiteBread/GRUS), my Sandhill Crane bioacoustics pipeline. Everything here is textbook methodology on openly licensed audio; the GRUS research pipeline and its findings are private pending peer review.

## What the notebook does

1. Loads audio from one of three sources (see below)
2. Renders waveform and spectrogram views
3. Detects call onsets with an energy-envelope method and estimates offsets by envelope decay, bounding each event by the next onset so neighbouring calls stay separate
4. Assigns a frequency band to each event from its band-limited energy
5. Exports a tab-separated **Raven selection table** and overlays the selections on the spectrogram

## Audio sources

The notebook tries three sources in order:

1. **Your own file.** Set `AUDIO_PATH` in the config cell to any local recording. This is the normal way to use it.
2. **xeno-canto.** As of 10 October 2025 the xeno-canto API (v3) requires a free API key. Register at [xeno-canto.org](https://xeno-canto.org), then set `XC_API_KEY` in your environment before launching Jupyter. No audio is redistributed in this repo; recordings are fetched at runtime and stay under the recordist's Creative Commons license.
3. **Synthesized fallback.** With neither of the above, the notebook generates a crane-like test signal so every cell still runs end to end. The committed outputs were produced this way, which is why the spectrogram looks synthetic.

## Run it

```bash
pip install -r requirements.txt
export XC_API_KEY=your_key_here      # optional
jupyter notebook crane_annotation_demo.ipynb
```

## Credit and licensing

Any xeno-canto audio remains under its recordist's Creative Commons license; the notebook prints the recording ID, recordist, and license when it fetches. Code in this repo is MIT.

---

Aaron Price · Anima Audire, LLC · [Profile](https://github.com/RobotWhiteBread) · aaron.price.unl@gmail.com
