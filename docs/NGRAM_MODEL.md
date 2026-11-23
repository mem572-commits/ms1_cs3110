# N-gram model

Owner: Nicholas Arana (nea43). Backup: Matheo Muecke (mem572).

This layer is the counting baseline for OcamlGPT. It takes the
`(context, target)` examples from `Corpus`, records how often each next
character id follows each context of `k` ids, and predicts the next id from
those counts. Its ids come from the same `Vocab` as `Neural`, so the two
models can be compared directly. The full spec is in `lib/ngram.mli`; this
page is the short version.

## Status

| Part | State |
|------|-------|
| `Ngram`: counting, prediction, generation | Implemented and tested |
| Average loss, interpolation for unseen contexts, evaluation script for validation loss | Proposed for MS3 (see the end of this page) |

## The Ngram module

| Module | What it is | Main functions |
|--------|------------|----------------|
| `Ngram` | A model that maps a fixed-length context of ids to next-id probabilities, built by counting | `fit`, `vocab_size`, `context_length`, `count`, `predict`, `sample_next`, `generate` |

### How the model works

With vocabulary size `V` and context length `k`:

1. **Context.** The last `k` ids, used as-is as a lookup key.
2. **Count.** `fit` makes one pass over the corpus and counts how many times
   each id follows each context of length `k`. There is no training loop.
3. **Prediction.** `p.(i) = count(ctx, i) / total count for ctx`, the
   relative frequency of each id after that context (`Ngram.predict`). An
   unseen context gives the uniform `1/V` for every id.
4. **Sampling.** `sample_next` picks a next id at random, with each id
   chosen with its probability in `predict`.
5. **Generation.** `generate` calls `sample_next` repeatedly. After each
   sample, the new id is appended to the context and the oldest one is
   dropped, so the window always holds the last `k` ids.

`k = 0` works too: every example shares the empty context, so the model
predicts from overall character frequencies, as a unigram model.

### Data flow

```
 text file ──Corpus.load──▶ Corpus.t ──Ngram.fit ~context:k──▶ Ngram.t
                                                                  │
                                                  Ngram.generate ~prompt ~length
                                                                  ▼
                                          Vocab.decode ◀── generated ids
```

`Ngram.fit` reads the examples itself (through `Corpus.examples_seq`), so
callers pass the corpus, not an example list.

## Rules other parts can rely on

- **Sizes come from `Corpus`.** `fit` reads `V` from `Corpus.vocab c`, so
  the model's ids always match `Vocab.decode (Corpus.vocab c)`. Contexts
  passed to `count`, `predict`, and `sample_next` must have length
  `context_length m`.
- **Two kinds of errors.** User-input problems (empty text, a negative
  context, a text no longer than the context) are `Corpus.error` results
  from `fit` and should be checked first. `Ngram` raises `Invalid_argument`
  only for caller mistakes: a context of the wrong length, an id out of
  range, a prompt shorter than the context, or a negative `length`.
- **A model is immutable after `fit`.** No function changes a fitted model,
  and `fit` is deterministic: the same corpus and `k` always give the same
  counts.
- **Seeded and reproducible.** The same `Random.State` gives the same
  generated text.
- **`generate` returns only the new ids**, not the prompt.

## How the CLI should call it (for Alejandro)

```ocaml
open Ocamlgpt

let run_ngram corpus ~context ~seed ~prompt ~length =
  match Ngram.fit corpus ~context with
  | Error e -> prerr_endline ("Error: " ^ Corpus.error_to_string e)
  | Ok model -> (
      match Corpus.encode_prompt corpus prompt with
      | Error e -> prerr_endline ("Error: " ^ Corpus.error_to_string e)
      | Ok ids when Array.length ids < context ->
          prerr_endline
            (Printf.sprintf "Error: the prompt needs at least %d characters." context)
      | Ok ids ->
          let out = Ngram.generate model (Random.State.make [| seed |]) ~prompt:ids ~length in
          print_endline (Vocab.decode (Corpus.vocab corpus) out))
```

Check the prompt length before calling `generate`, as above, so a short
prompt gives a clear message instead of an `Invalid_argument` exception.

## Design choices

- **A `Hashtbl` keyed by context arrays.** The model stores only the
  contexts that were seen, so memory grows with the text, not with `V^k`.
  A lookup is one hash of `k` ids.
- **Counts are stored, not probabilities.** `count` comes for free,
  `predict` divides at query time, and interpolation and smoothing in MS3
  need the raw counts.
- **An unseen context gives uniform output, not an error.** An unseen
  context has no counts to divide, so `predict` needs a fallback to return
  a valid distribution. Uniform is the simplest one that never assigns zero
  to any id. MS3 replaces it with interpolation (below).
- **The model is separate from the CLI.** Everything here can be tested
  without running the program.

## Validation

### Manual check on `data/sample.txt`

`Ngram.generate` on `data/sample.txt` with `~context:3`, prompt `"the"`,
seed `[| 3110 |]` and `~length:200` gave:

```
 lazy dog from the dog fox runs the quick brown fox and the quick fox runs to the fox runs the dog fox and the fox jumps in the dog.
the fox jumps in the fox and the old stone wall cat wall.
fbpcqvtuk
```

Up to `wall.\n`, every three-character window appears in the training
text, so the output is made of fragments of it. Then generation reaches the
context `l.\n`, which appears only at the very end of the file and so never
has a recorded successor. That context gives a uniform distribution, the
random character it picks forms another unseen context, and the output
never recovers. This is the main weakness of the model, and the motivation
for interpolation in MS3.

### Done (all pass: `dune test`, file `test/test_ngram.ml`, 37 tests)

The expected values were worked out by hand from the fixture strings
(`"abcabd"` and `"ababab"`), which are noted in comments above the tests.

| Area | What is checked | Why |
|------|-----------------|-----|
| Accessors (3) | `vocab_size` is the number of distinct characters; `context_length` is the fitted `k`, including `k = 0` | Every other function relies on these sizes |
| Fit (3) | Succeeds at `length = k + 1`; `Too_short` at `length = k` and beyond | The boundary is where off-by-one mistakes hide |
| Counts (5) | `ab → c` and `ab → d` once, `ab → a` never; an unseen context gives 0; `k = 0` counts every occurrence | Counting is the whole model; the answers come from reading the text by hand |
| Predict (4) | After `ab`, exactly `[0; 0; 0.5; 0.5]`; sums to 1; an unseen context is uniform; `k = 0` gives `[1/3; 1/3; 1/6; 1/6]` | Checks the actual distribution, not just its shape, so a model that ignored its counts would fail |
| `sample_next` (4) | A forced context returns the same id for every seed; results are in range; after `ab`, only `c` and `d` appear, each about half the time | Catches off-by-one errors in the cumulative sum |
| Generate (6) | Output length; length 0 gives `[||]`; the same seed gives the same output; ids in range; a long prompt uses only its last `k` ids; `k = 0` accepts an empty prompt | Generation must never produce an invalid id, and the sliding window must use the right ids |
| Properties (3 × 300 random texts and `k`) | `predict` sums to 1 on every seen context; the counts add up to `length - k`; `generate` gives in-range ids of the right length | Catches dropped or double-counted examples and corrupted keys across inputs no hand-written case covers |
| Errors (9) | `fit` rejects a negative context; `count`, `predict`, and `generate` raise `Invalid_argument` for each bad context, id, prompt, or length | Caller mistakes must fail loudly instead of returning a wrong answer |

### Planned (not yet checked)

- **Validation loss against `Neural`.** On a training file of tens of
  kilobytes, split with `Corpus.split`, compare `Ngram.average_loss` with
  `Neural.average_loss` on the validation part for the same `k`.
- **Interpolation.** Check that the blended distribution still sums to 1,
  and that no id ever gets probability 0, which is the property that keeps
  the validation loss finite.

## Proposed for MS3 (not implemented)

```ocaml
(* Mean of -. log p.(target) over the examples, to compare with
   Neural.average_loss *)
val Ngram.average_loss : t -> Corpus.example list -> float

(* Interpolated prediction: blends every context length from k down to 0 *)
val Ngram.predict_interpolated : t -> lambdas:float array -> int array -> Matrix.vec
```

- **Average loss.** With the current model, this would be infinite on
  most validation text: a context seen in training but followed there only
  by other ids gives the true target probability 0, and `-. log 0.` is
  infinity. So average loss is only a useful comparison once interpolation
  is in place.
- **Interpolation for unseen contexts.** Always blend every context length:

  ```
  p(c | abc) = λ₃·p̂(c | abc) + λ₂·p̂(c | bc) + λ₁·p̂(c | c) + λ₀·p̂(c)
  ```

  The λ weights sum to 1. Each p̂ is the raw count divided by the total for
  that context length; if a context wasn't seen, drop that term and
  renormalize the remaining λs. Add add-k smoothing to the bottom term p̂(c)
  so no id is ever 0. Try several λ settings and pick the one with the
  lowest validation loss, using `Corpus.split` for the train/validation
  split. This needs counts for every length from `k` down to 0, so `fit`
  would store one table per length.
- **Evaluation script.** A `bench/ngram_eval.ml` that fits on the training
  split and prints the validation loss for each `k` and λ setting, as
  `bench/neural_bench.ml` does for the neural model.
- **Temperature for sampling.** Raise each `p.(i)` to the power `1/T` and
  renormalize before sampling: `T < 1` sharpens toward the most likely id,
  and `T > 1` flattens toward uniform.

These signatures are a starting point for discussion with Matheo (neural
model) and Alejandro (CLI), so both models can be chosen and compared the
same way.
