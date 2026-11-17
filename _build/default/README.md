# OcamlGPT

A command-line tool for training small character-level language models,
built from scratch in OCaml. CS 3110, Group 60.

## Team

- Matt Zhu (hz754)
- Matheo Muecke (mem572)
- Nicholas Arana (nea43)
- Alejandro Lilienthal (al2594)

## Build and run

Requires OCaml 5 with `dune`, `ounit2`, and `qcheck` (all in the CS 3110 opam
switch).

```bash
dune build
dune test
dune exec ocamlgpt -- data/sample.txt 3
```

The last command loads a text file and reports its vocabulary and the number
of training examples for a context length of 3.

## Layout

```
lib/      library (Ocamlgpt)
  vocab   character vocabulary: encode / decode
  corpus  file loading, validation, train/validation split, training examples
bin/      command-line entry point (placeholder)
test/     unit and property tests, plus test/fixtures/
data/     sample training text
docs/     notes for teammates (start with docs/TEXT_PROCESSING.md)
```
