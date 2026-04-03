# Continuous Constraints, Empirical Density, and Violation Computation

This repository accompanies the paper **“Continuous Constraints, Empirical Density, and Violation Computation.”** It contains the manuscript source, the illustrative corpus used by the implementation, the computational script that carries out the analysis, and a graphical interface for interactive use. It also contains two zip bundles: "plots.zip" (all images generated from the script) and "text_reports.zip" (the text file reports generated from the script).

The repository is intended as a reproducible companion to the paper rather than as a general-purpose phonological software package. Its principal purpose is to make the computational workflow described in the article inspectable and rerunnable over the illustrative data, while also allowing readers to substitute their own datasets and explore alternative continuous candidate regions.

## Repository contents

- `main.tex` and `section_*.tex`: LaTeX source of the paper.
- `main.pdf`: compiled paper.
- `app.py`: computational implementation.
- `ui.py`: graphical interface for launching the analysis.
- `data.txt`: illustrative acoustic dataset.

## Quick start

For complete reproducibility of the paper, the recommended procedure is to launch the graphical interface and keep the default values unchanged.

1. Place `app.py`, `ui.py`, and `data.txt` in the same directory.
2. Install the required Python dependencies.
3. Run:

```bash
python3 ui.py
```

4. In the interface, leave the default values unchanged:
   - `Speakers = 1, 3, 5`
   - `Vowel = e`
   - `Weight [N] = 1.027`
   - `Weight [CP] = 0.417`
   - `Weight [CA] = 0.018`
5. Ensure that the checkbox **“Calculate weights?”** remains unchecked.
6. Press **Start**.

Under those settings, the interface reproduces the subset used in the paper and uses the weights already estimated for that subset.

## Reproducibility and the corpus subset used in the paper

A crucial point for reproducibility is that `data.txt` contains **more speakers and more vowels** than the subset used in the paper. The article deliberately uses only a restricted portion of the corpus for demonstration and testing. The worked example in the paper is based on:

- only one vowel category: `e`
- only three speakers: `1, 3, 5`
- the specific subset identified in the paper as the male Northeastern speakers selected for the demonstration

Accordingly, the default values shown when `ui.py` is launched are not arbitrary convenience settings. They are the reproducibility settings for the paper.

The default weights in the interface,

- `Weight [N] = 1.027`
- `Weight [CP] = 0.417`
- `Weight [CA] = 0.018`

are the weights previously estimated for that exact subset through **L-BFGS-B** optimization. If the user keeps the default speakers, vowel, candidate bounds, and weight values exactly as provided, the run is configured to reproduce the demonstration reported in the paper.

By contrast, if the user changes the corpus selection or substitutes a new dataset, those default weights should no longer be treated as theoretically appropriate for the modified data. In such cases, the checkbox **“Calculate weights?”** should be enabled so that the program re-estimates the weights by optimization. This procedure may take a substantial amount of time, because the implementation repeatedly constructs continuous model densities and compares them to the empirical density during the search for the best-fitting parameter values.

## Running the implementation

### Graphical interface

The principal entry point for most readers is:

```bash
python3 ui.py
```

The interface allows the user to:

- select the input data file;
- specify the subset of speakers;
- specify the vowel category or categories;
- define the candidate interval in F1 and F2;
- set the constraint-related parameters;
- either supply weights directly or request weight optimization;
- execute the computational pipeline and inspect the running log in the interface window.

### Direct execution of the computational script

The computational script may also be run directly:

```bash
python3 app.py
```

When executed without explicit command-line arguments, it falls back to the same default reproducibility settings used by the interface. For routine use, however, the interface is the clearer and safer mode of operation, since it makes the current settings visible before execution.

## Software requirements

The code is written for Python 3. The implementation relies on standard scientific Python libraries for numerical computation, optimization, plotting, interpolation, and density estimation, together with Tk for the graphical interface.

A typical installation will require packages such as:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pillow sympy tqdm scienceplots
```

Notes:

- `scienceplots` is stylistic rather than essential. If it is unavailable, the plotting system falls back to a default style.
- `tkinter` is used for the graphical interface. On some systems it is bundled with Python; on others it must be installed separately through the operating system package manager.

## Structure of `data.txt`

The illustrative dataset is a plain text table with whitespace-separated columns and a header row. The expected structure is:

```text
Falante Vogal F1 F2
1 e 424.911 1523.522
1 e 440.760 1391.502
...
```

The columns have the following interpretation:

- `Falante`: speaker identifier
- `Vogal`: vowel label
- `F1`: first formant value
- `F2`: second formant value

The implementation expects at least these four columns, with these header names, and expects the file to be readable as a whitespace-delimited table. The analysis filters the dataset by speaker and vowel before constructing the empirical continuous candidate space. Rows whose `F1` or `F2` values are marked as `NA` are ignored.

Users who wish to provide their own dataset should therefore preserve the same general organization:

1. one row per token;
2. one speaker identifier column;
3. one vowel label column;
4. numeric `F1` and `F2` columns;
5. a header row;
6. whitespace-separated formatting.

The implementation is designed for acoustic data expressed in two continuous dimensions, here F1 and F2. If a new dataset follows the same structural format, it can be used in place of `data.txt`.

## Conceptual overview of the computational implementation

The computational architecture follows the central methodological claim of the paper: continuous phonological evaluation should not begin from a pre-binned candidate list. Instead, it begins from an empirical cloud of observations in continuous acoustic space and computes evaluation over that space directly.

At a high level, the implementation proceeds through the following stages.

### 1. Corpus selection

The program first selects a subset of tokens from the input dataset according to the chosen speakers and vowel labels. In the reproducibility configuration of the paper, this means selecting the tokens for speakers `1, 3, 5` and vowel `e`.

### 2. Continuous representation of the empirical candidate space

The selected tokens are then represented as a continuous empirical distribution over the F1–F2 space. In conceptual terms, the corpus is not treated as a finite set of isolated points or as a set of pre-defined bins. Instead, it is converted into a smooth density surface that represents where the observed data are concentrated.

This stage corresponds directly to the paper’s argument that empirical density should be treated as the corpus-conditioned representation of candidate availability rather than as a constraint or as a substitute for grammar.

### 3. Standardization

Before density estimation and evaluation, the selected acoustic values are standardized so that the dimensions can be handled on a common numerical scale. This is a computational step for stable estimation and comparison across the two dimensions; it is not, by itself, a theoretical claim about the grammar.

### 4. Continuous constraint evaluation

The implementation then evaluates two proof-of-concept continuous constraints over the candidate space:

- a perceptual constraint, evaluated over the F1 dimension;
- an articulatory constraint, evaluated over the joint F1–F2 space.

In keeping with the paper, these are not used as a final universal inventory of phonological constraints. They serve as explicit continuous constraint families through which the architecture can be demonstrated computationally.

### 5. Continuous violation computation

The key step is the construction of continuous violations from the interaction between the constraint surfaces and the empirical density. The implementation follows the architecture defended in the paper: violations are not assigned to pre-delimited candidates, but accumulated from local interactions across continuous space.

For this reason, the candidate region specified by the user through `F1 (min.)`, `F1 (max.)`, `F2 (min.)`, and `F2 (max.)` should be understood as a **region of integration and reporting**, not as a discrete pre-evaluative bin in the classical sense. The underlying evaluative object remains continuous.

### 6. Weighted evaluation

Once continuous violation quantities have been obtained, they are weighted. In the reproducibility configuration, the weights are the pre-estimated values supplied in the interface. In optimization mode, the weights are estimated from the selected dataset.

The weighted sum yields a harmonic score over the candidate region. In interpretive terms, this quantity summarizes the total weighted pressure associated with the selected region under the current empirical density and constraint settings.

### 7. Maximum-entropy model density

The implementation next constructs a continuous maximum-entropy-style output density from the weighted violations. Conceptually, this means that the model does not merely compute violation magnitudes; it also defines a continuous probabilistic distribution over outputs.

This stage corresponds to the paper’s broader point that the same architecture can support not only continuous violation computation but also continuous probabilistic evaluation.

### 8. Model–data comparison

The modeled density is then compared to the empirical density. The implementation reports a Kullback–Leibler divergence value, which quantifies the discrepancy between the model distribution and the empirical distribution. In practical terms, lower divergence indicates a closer fit between the modeled output density and the observed data distribution.

### 9. Graphical and textual reporting

Finally, the program writes a report and generates figures that summarize the empirical density, the modeled density, the candidate-region probabilities, and several views of how violations develop across the continuous space.

## Meaning of the main outputs

A typical run produces a set of text files and figures in the working directory. The precise collection is oriented toward inspection rather than minimalism, since the repository accompanies a paper and is meant to expose the internal behavior of the implementation.

### `report.txt`

This is the principal textual summary of a run. It records:

- the input file used;
- the selected vowel(s);
- the selected speaker(s);
- the candidate region in F1 and F2;
- the constraint weights used in the run;
- perceptual and articulatory violation totals for the selected region;
- the harmonic score;
- the stability score;
- the F1 probability, F2 probability, and joint F1–F2 probability for the selected region under the modeled distribution;
- the Kullback–Leibler divergence between model and data.

This file is the most convenient summary for comparison across runs.

### `sample_values.txt`

This file contains sampled values from the empirical probability density over the continuous candidate space. It is useful for inspecting the estimated empirical distribution numerically rather than only graphically.

### Density and model comparison figures

The implementation generates figures that display:

- the empirical density alone;
- the modeled MaxEnt density alone;
- a direct comparison between empirical and modeled marginals along F1 and F2;
- divergence-oriented comparisons between model and data.

These figures make it possible to inspect how well the model tracks the empirical distribution in each dimension.

### Violation and progression figures

Additional figures visualize:

- the relation between empirical density, constraint shape, and resulting violations;
- the progressive accumulation of violations across the continuous space;
- comparisons between articulatory violations computed with density-sensitive evaluation and corresponding views without density modulation;
- harmonic-score-oriented views associated with the articulatory component.

These figures are especially helpful for understanding the paper’s central claim that the same constraint family can yield different violation distributions when the empirical density changes.

## Interpreting the reported quantities

The implementation reports several summary quantities whose interpretation is easiest when kept at a conceptual level.

### Violation totals

These are integrated quantities over the selected candidate region. They summarize the amount of perceptual and articulatory violation accumulated over that region under the current density-sensitive continuous evaluation.

### Harmonic score

This is the weighted sum of the constraint-specific violation quantities. It is the continuous analogue of a weighted grammatical evaluation over the selected region.

### Stability

This quantity summarizes how sharply the weighted evaluative surface changes over the selected candidate region. It is best treated as a diagnostic of the shape of the evaluative landscape rather than as a stand-alone linguistic claim.

### Candidate-region probabilities

The reported F1, F2, and joint probabilities indicate how much modeled probability mass falls within the selected candidate interval or region. They therefore connect the continuous evaluative architecture to a probabilistic interpretation of candidate regions.

### Kullback–Leibler divergence

This is the model-fit quantity used in the implementation. It compares the modeled output density with the empirical density inferred from the data. In optimization mode, the weight estimation procedure attempts to reduce this divergence.

## Exploring different continuous candidates

One of the principal advantages of the implementation is that the user can change the candidate bounds and immediately inspect the consequences. In the interface, this is done through:

- `F1 (min.)`
- `F1 (max.)`
- `F2 (min.)`
- `F2 (max.)`

Changing these values does not redefine the entire system as a discrete binning procedure. Rather, it changes the region over which continuous quantities are summarized and reported. This allows readers to inspect broader or narrower candidate windows while preserving the same underlying continuous evaluative architecture.

As these bounds are changed, the generated plots and the contents of `report.txt` will change accordingly. This makes the repository useful not only for reproducing the paper but also for exploring how different candidate regions behave under the same general model.

## Using your own data

Readers may use their own corpus instead of `data.txt`, provided that the new file follows the same general format and supplies speaker labels, vowel labels, and numeric F1/F2 measurements.

When using a new dataset, the following procedure is recommended:

1. format the dataset with the same four columns (`Falante`, `Vogal`, `F1`, `F2`);
2. select the file in the interface;
3. choose the desired speakers and vowel labels;
4. set the candidate bounds;
5. enable **“Calculate weights?”**;
6. run the analysis and allow the optimizer to estimate new weights.

This is important because the default weights in the interface were estimated for the paper’s demonstrative subset, not for arbitrary corpora.

## Scope and intended use

This repository should be read in the same spirit as the paper itself. It is a worked computational demonstration of a methodological architecture for continuous-candidate phonological evaluation. It is not presented as a finished grammar of Brazilian Portuguese pretonic vowels, nor as a final-purpose software library. Its principal value lies in making the paper’s claims computationally explicit, runnable, and inspectable.
