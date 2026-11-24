# OcamlGPT

A command-line program for training small character-level language models
from scratch in OCaml. You give it a plain-text file, it fits a model on that
text, and it generates new text from a starting prompt.

Right now the program uses a character n-gram model. A one-layer neural model
is implemented in the library (`lib/neural.ml`) but is not connected to the
command line yet.

Authors:

Matheo Muecke (mem572)
Matt Zhu (hz754)
Nicholas Arana (nea43)
Alejandro Lilienthal-Gerson (al2594)

## Setup

Needs OCaml 5 and dune. Install the test libraries with:

```bash
opam install ounit2 qcheck-core qcheck-ounit
```

Then build from the repository root:

```bash
dune build
```

## Running

```bash
dune exec ocamlgpt -- <training-file> [-k CONTEXT] [-n LENGTH] [-seed SEED] [-prompt TEXT]
```

| Option | Meaning | Default |
|--------|---------|---------|
| `-k` | Context length: how many previous characters the model looks at | 3 |
| `-n` | Number of characters to generate | 200 |
| `-seed` | Random seed. The same seed always gives the same text | 3110 |
| `-prompt` | Starting text. Must be at least `-k` characters, using only characters from the training file | The first `-k` characters of the file |

Example:

```bash
dune exec ocamlgpt -- data/sample.txt -k 3 -n 200 -seed 3110 -prompt "the"
```

prints a line with the settings, then the prompt followed by 200 generated
characters:

```
n-gram model: context 3, vocabulary 29, seed 3110

the lazy dog from the dog fox runs the quick brown fox and the quick fox runs to the fox runs the dog fox and the fox jumps in the dog.
the fox jumps in the fox and the old stone wall cat wall.
fbpcqvtuk
```

The random characters at the end happen when the model reaches a context it
never saw in training (see `docs/NGRAM_MODEL.md`).

If something is wrong with the input, such as a missing file, a prompt that is
too short, or a character the training text never uses, the program prints
`Error: ...` and exits with code 1. A missing training file or a bad option
exits with code 2. `-help` lists the options.

## Testing

```bash
dune test
```

runs every test suite:

| File | What it tests |
|------|---------------|
| `test/test_ocamlgpt.ml` | Text processing: `Vocab` and `Corpus` |
| `test/test_ngram.ml` | The n-gram model: counts, probabilities, sampling, generation |
| `test/test_neural.ml` | `Matrix` and the neural model |
| `test/test_cli.ml` | End to end: runs the `ocamlgpt` program and checks its output and exit code |

The neural model benchmark is separate:

```bash
dune exec bench/neural_bench.exe
```

## Design docs

- `docs/TEXT_PROCESSING.md`: `Vocab` and `Corpus`
- `docs/NGRAM_MODEL.md`: the n-gram model
- `docs/NEURAL_MODEL.md`: `Matrix` and the neural model
