# Neural model and numerical operations

Owner: Matheo Muecke (mem572). Backup: Alejandro Lilienthal (al2594).

This layer is the first neural language model in OcamlGPT. It takes the
`(context, target)` examples from `Corpus`, learns to predict the next
character id, and generates new ids that `Vocab` turns back into text. The
full specs are in `lib/matrix.mli` and `lib/neural.mli`; this page is the
short version.

## Status

| Part | State |
|------|-------|
| `Matrix`: vector and matrix operations | Implemented and tested |
| `Neural`: one-layer model, training, generation | Implemented and tested |
| `bench/neural_bench.ml`: speed and training investigation | Implemented, results below |
| Hidden-layer model, save/load, CLI integration | Proposed for MS3 (see the end of this page) |

## The two modules

| Module | What it is | Main functions |
|--------|------------|----------------|
| `Matrix` | `vec = float array`, `mat = float array array` (row by row) | `zeros`, `random`, `one_hot`, `mat_vec`, `mat_mul`, `add_vec`, `dot`, `softmax`, `argmax` |
| `Neural` | A model that maps a fixed-length context of ids to next-id probabilities | `create`, `of_params`, `predict`, `example_loss`, `average_loss`, `train_step`, `train`, `sample_next`, `generate` |

### How the model works

With vocabulary size `V` and context length `k`:

1. **Input.** Each of the `k` context ids becomes a one-hot vector of length
   `V`. They are joined end to end into `x`, of length `k * V`
   (`Neural.input_of_context`).
2. **Scores.** `W x + b`, where `W` is `V × (k*V)` and `b` has length `V`.
3. **Prediction.** `p = softmax (W x + b)`: one probability per next id
   (`Neural.predict`).
4. **Loss.** `-log p.(target)` (`Neural.example_loss`). A uniform guess
   scores `ln V`.
5. **Training.** Stochastic gradient descent, one example at a time
   (`Neural.train_step`). The gradient of score `j` is `p.(j) - 1` for the
   target and `p.(j)` otherwise. The weight gradient multiplies that by
   `x.(k)`.

`k = 0` works too: then the model predicts from the bias alone, as a
unigram model.

### Data flow

```
 text file ──Corpus.load──▶ Corpus.t ──Corpus.examples ~context:k──▶ example list
                                                                          │
                                           Neural.create ~vocab_size ~context
                                                                          ▼
                     Vocab.decode ◀── Neural.generate ◀── Neural.train (updates weights)
```

## Rules other parts can rely on

- **Sizes come from `Corpus`.** Use `Vocab.size (Corpus.vocab c)` for
  `~vocab_size`. Use the same `k` for `Corpus.examples ~context:k` and
  `Neural.create ~context:k`.
- **Two kinds of errors.** User-input problems (missing file, empty text,
  bad context length) are `Corpus.error` results and should be checked
  first. `Matrix` and `Neural` raise `Invalid_argument` only for caller
  mistakes: wrong context length, an id out of range, a prompt shorter than
  the context, or mismatched sizes.
- **Only training changes a model.** `train_step` and `train` update the
  weights in place. Everything else, including `weights` and `bias`
  (which return copies), leaves the model unchanged.
- **Seeded and reproducible.** The same seed gives the same initial model,
  and the same `Random.State` gives the same generated text.
- **`generate` returns only the new ids**, not the prompt.

## How the CLI should call it (for Alejandro)

```ocaml
open Ocamlgpt

let run_neural corpus ~context ~epochs ~length =
  match Corpus.examples corpus ~context with
  | Error e -> prerr_endline ("Error: " ^ Corpus.error_to_string e)
  | Ok exs ->
      let vocab = Corpus.vocab corpus in
      let model = Neural.create ~vocab_size:(Vocab.size vocab) ~context ~seed:3110 in
      let losses = Neural.train model ~learning_rate:0.5 ~epochs exs in
      Printf.printf "final loss %.3f\n" (List.nth losses (epochs - 1));
      (* the prompt must have at least [context] characters *)
      let prompt = Array.sub (Corpus.ids corpus) 0 context in
      let ids = Neural.generate model (Random.State.make [| 3110 |]) ~prompt ~length in
      print_endline (Vocab.decode vocab ids)
```

For a user-typed prompt, use `Corpus.encode_prompt` and check that the
result has at least `context` ids before calling `generate`.

## Design choices

- **The matrix representation is plain and visible.** `float array array`
  is easy to read, test, and print. The model hides its own weights behind
  the abstract `Neural.t`, so other code cannot change them without going
  through `train_step`.
- **Weights are updated in place.** Copying a `V × kV` matrix on every
  example would be slow and add nothing for a proof of concept. The specs
  say exactly which functions change a model.
- **`of_params` builds a model from exact weights.** Tests use it to check
  predictions against hand-computed values. It is also the hook for loading
  saved weights later.
- **The model is separate from the CLI.** Everything here can be tested
  without running the program.

## What the investigation found (`bench/neural_bench.ml`)

Run with `dune exec bench/neural_bench.exe`. These are the results from the
MS2 run (timings are CPU time and will vary by machine):

| Check | Result |
|-------|--------|
| `mat_mul`, n = 50 / 100 / 200 / 400 | 0.0002 s / 0.0016 s / 0.017 s / 0.105 s (about 8× per doubling, as expected for n³) |
| 100 epochs on `data/sample.txt` (context 3, V = 29, 172 examples) | 0.107 s in total, about 0.001 s per epoch |
| Training loss | 3.367 (= ln 29) → 0.387 after 10 epochs → 0.279 after 100 |
| Validation loss (last 10% of the text) | 3.366 → 3.837 → 5.500, rising throughout |
| Generated text | `the dog from fox jumps in the river and the dog from llecps in the dat wathe fox runs …` |

What we learned:

- **Speed is not the bottleneck.** Plain OCaml arrays train this model in
  milliseconds, so the MS1 worry about numerical speed does not block the
  neural model.
- **The sample text is far too small.** The training loss falls while the
  validation loss rises, which means the model is memorizing 195
  characters rather than learning patterns that carry over to unseen text.
  Before comparing the n-gram and neural models, we need a larger training
  file (tens of kilobytes) and should report validation loss, stopping
  when it stops improving.

## Validation

### Done (all pass: `dune test`, file `test/test_neural.ml`, 52 tests)

The expected values were worked out by hand and are written as comments
next to each test.

| Area | What is checked | Why |
|------|-----------------|-----|
| Matrix (22 tests + 1 property) | `dot`, `mat_vec` (2×3), `mat_mul` (2×2) against hand results; size mismatches raise; `softmax [0; ln 3] = [1/4; 3/4]`; `softmax [1000; 1000]` does not overflow; ties in `argmax`; a property (300 random vectors) that softmax is positive and sums to 1 | An indexing or overflow mistake here would break every model |
| Model shape and input (10) | Sizes from `create`, seeding, copies not shared, one-hot layout, bad context or id raises, context 0 | A wrong layout would silently mix up context positions |
| Exact predictions (8) | Hand-set weights give `p = [1/4; 3/4]` and loss `ln(4/3)`; a new model's loss is about `ln V` | Checks the forward pass without trusting training |
| Gradient check (1) | For every weight and bias, the update from `train_step` matches a central finite difference of the loss to within 1e-6 | Catches sign and index mistakes in the hand-derived gradient |
| Learning (4) | On `abcabcabd` with context 2, after training, p(c\|ab) ≈ 2/3 and p(d\|ab) ≈ 1/3 (within 0.1), and p(a\|bc) > 0.9; loss decreases; 0 epochs changes nothing | The right answers come from counting the text by hand, independent of the code |
| Generation (5) | Output length, ids in range, same seed gives the same output, short prompt raises, a certain prediction is always sampled | Generation must never produce an invalid id |
| Integration (1) | `Corpus.load` → `Corpus.examples` → `Neural.train` → `Neural.generate` → `Vocab.decode` gives text using only corpus characters | Shows the components connect |

Manual checks done with `bench/neural_bench.exe`: the normal run (results
above); `-- nope.txt` prints `Error: File not found: nope.txt` and exits 1;
`-- test/fixtures/empty.txt` prints `Error: The training text is empty.`
and exits 1.

### Planned (not yet checked)

- **Larger corpus.** Train on a file of tens of kilobytes and check that the
  validation loss falls below `ln V` and below the n-gram model's
  validation loss for the same context length.
- **Hidden layer.** Repeat the finite-difference gradient check for the
  new layer, and check that it learns XOR-like patterns the one-layer
  model cannot.
- **Save/load.** Check that saving and then loading a model gives identical
  `predict` output on every context of a test corpus, and that loading a
  file with the wrong vocabulary is rejected.
- **CLI.** Manual check that choosing the neural model prints falling loss
  and generated text, and that a prompt shorter than the context gives a
  clear message instead of an exception.
- **Coverage.** Run Bisect (`bisect_ppx` is installed) on `lib/matrix.ml`
  and `lib/neural.ml`, and add tests for any branch the suite never reaches.

## Proposed for MS3 (not implemented)

```ocaml
(* Hidden-layer model: x -> tanh (W1 x + b1) -> softmax (W2 h + b2) *)
val Neural.create_mlp : vocab_size:int -> context:int -> hidden:int -> seed:int -> t

(* Save and load, storing Vocab.serialize alongside the weights *)
val Neural.save : t -> Vocab.t -> string -> (unit, string) result
val Neural.load : string -> (t * Vocab.t, string) result
```

These signatures are a starting point for discussion with Alejandro (CLI)
and Nicholas (n-gram model), so both models can be chosen and compared the
same way.
