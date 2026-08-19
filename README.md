# crane-annotation-demo

**A bioacoustic annotation workflow on real field audio, including where it breaks.**

This notebook walks the basic path every annotation project shares: load a recording, view it as a spectrogram, detect call boundaries, and export a selection table an expert can open and correct by hand.

It also does the thing demo notebooks usually skip, which is to show where the method stops working. The same roost, nineteen minutes later, defeats it completely. Being able to say precisely when a detector has stopped being informative is worth more than the detector.

Companion to the private GRUS research pipeline. Everything here is textbook methodology; none of it is part of that pipeline.

## The audio

Two clips recorded by the author on the central Platte River, Nebraska, at dawn in March 2026. Both come from one continuous recording, about nineteen minutes apart.

| clip | separability | what it is |
|---|---|---|
| `crane_roost_sparse_30s.wav` | 2.69 | early, before the roost fully wakes; calls are separable |
| `crane_roost_chorus_20s.wav` | 1.35 | later, near liftoff; a continuous wall of sound |

Separability is peak band energy over median band energy. A ratio near 1 means the loudest moment is barely louder than a typical one.

## What the notebook does

1. Looks at the spectrogram before running anything
2. Shows that stock `librosa` onset settings fire **7.4 times per second** on this material, tracking chorus ripple rather than birds
3. Sweeps `delta` and `wait`, finding that usable behaviour occupies a narrow band with no stable plateau, and documents the chosen values as chosen rather than derived
4. Detects 15 **calling bouts** in 30 seconds, and is explicit that these are not individual calls
5. Cross-checks with an independent spectral-flux detector, since there are no hand labels and agreement between methods that fail differently is the available evidence
6. Exports a Raven-compatible selection table with a `Corroborated` column
7. Runs the same pipeline on the chorus clip, where it finds **zero** events

## The result worth reading

On the dense clip the detector returns nothing at all. Downstream, zero detections reports as *no calling activity* at the exact moment thousands of birds are calling at once. Nothing raises, nothing logs a warning, and the output is structurally indistinguishable from a genuinely quiet recording.

That is the argument for computing a separability statistic and reporting it next to the detections, so an empty result can be told apart from an uninformative one.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook crane_annotation_demo.ipynb
```

The audio ships with the repo, so it runs with no setup and no API keys. `make_notebook.py` regenerates the notebook if you would rather not hand-edit JSON.

## What this is not

No learned models, no call-type classification, no individual identification, no source separation. Production annotation work, meaning multi-observer protocols, call-type taxonomies, and sequence analysis, lives in the private GRUS pipeline pending publication.

## Licensing

**Audio** © 2026 Aaron Price, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse it, credit the recordist.

**Code** MIT.

Raven is named only to describe the output format. This repository bundles, links, and requires no Raven software and is not affiliated with or endorsed by the Cornell Lab of Ornithology. [Raven Pro and Raven Lite](https://www.ravensoundsoftware.com/) carry their own licenses.

---

Aaron Price · Anima Audire, LLC · [Profile](https://github.com/RobotWhiteBread) · aaron.price.unl@gmail.com
