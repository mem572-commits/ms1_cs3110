# Text processing and dataset handling

Owner: Matt Zhu (hz754). Backup: Nicholas Arana (nea43).

This layer turns a text file into numbers that models can use, and turns
model output back into text. The full specs are in `lib/vocab.mli` and
`lib/corpus.mli`; this page is the short version.

## The two modules

| Module   | What it is | Main functions |
|----------|------------|----------------|
| `Vocab`  | A mapping between characters and ids `0 .. size - 1` | `of_string`, `size`, `encode`, `decode`, `try_encode`, `char_of_id`, `id_of_char`, `serialize` |
| `Corpus` | A training text plus its `Vocab` and its ids | `load`, `of_string`, `vocab`, `ids`, `examples`, `examples_seq`, `split`, `encode_prompt`, `summary`, `error_to_string` |

Everything is immutable. Anything that can fail because of user input returns
`(_, Corpus.error) result`. Turn the error into a message with
`Corpus.error_to_string`.

## Rules every part can rely on

- **Ids are stable.** Ids follow character-code order (`'\n'` < `' '` < `'.'` <
  `'a'` …), so the same characters always get the same ids.
- **Line endings are normalized.** `Corpus.load` turns `\r\n` and `\r` into `\n`
  and removes a UTF-8 BOM, so a file gives the same result on Windows and on
  macOS/Linux. `Corpus.of_string` uses its text exactly as given.
- **A character is a byte.** ASCII works as expected. UTF-8 text still round-trips
  exactly, but `é` counts as two "characters".
- **A corpus is never empty.** `Corpus.length c >= 1` always holds.
- **Splits share one vocabulary.** Both halves of `Corpus.split` use the full
  corpus's vocabulary, so their ids agree.

## How each part uses it

### N-gram model (Nicholas)

```ocaml
open Ocamlgpt

let count_ngrams corpus ~n =
  (* n = context length: number of characters before the predicted one *)
  match Corpus.examples_seq corpus ~context:n with
  | Error e -> Error e
  | Ok examples ->
      let table = Hashtbl.create 1024 in
      Seq.iter
        (fun { Corpus.context; target } ->
          (* context : int array of length n, target : int *)
          ...)
        examples;
      Ok table
```

- `Vocab.size (Corpus.vocab corpus)` is the number of possible next characters.
  Use it to size probability arrays.
- `int array` works as a `Hashtbl` key.
- `~context:0` gives a unigram model (an empty context for every character).
- After sampling an id, turn it into text with `Vocab.decode_id vocab id`, or a
  whole array with `Vocab.decode vocab ids`.

### Neural model (Matheo)

- The same `Corpus.examples` gives `(context, target)` pairs for a fixed-length
  context. One-hot vectors have length `Vocab.size`.
- `Corpus.split corpus ~train_ratio:0.9` gives training and validation corpora
  for reporting loss.
- To save a model, store `Vocab.serialize vocab` with the parameters. Restore it
  with `Vocab.of_string`.

### Command-line interface (Alejandro)

```ocaml
match Corpus.load path with
| Error e -> prerr_endline ("Error: " ^ Corpus.error_to_string e); exit 1
| Ok corpus ->
    print_endline (Corpus.summary corpus);
    (match Corpus.encode_prompt corpus prompt with
     | Error e -> prerr_endline ("Error: " ^ Corpus.error_to_string e)
     | Ok prompt_ids -> (* pass prompt_ids to the model *) ...)
```

Errors the CLI should expect, with the message each one prints:

| Error | Cause |
|-------|-------|
| `File_not_found p` | No file at `p` |
| `Not_a_file p` | `p` is a directory |
| `Read_error (p, msg)` | File exists but cannot be read |
| `Empty_text` | File is empty |
| `Too_short {length; required}` | Text shorter than context length + 1, or too short to split |
| `Invalid_context_length n` | Negative context length |
| `Invalid_ratio r` | Train ratio not strictly between 0 and 1 |
| `Unknown_chars cs` | Prompt (or held-out text) uses characters the training text never has |

`bin/main.ml` is a placeholder that shows these calls working. Replace it with
the real CLI.

## Test fixtures

`test/fixtures/` holds small files for tests. Each part can use them in its own
tests. The test `dune` file already copies them next to the test executable, so
open them as `"fixtures/<name>"`.

| File | Contents |
|------|----------|
| `hello.txt` | `hello\n` |
| `abc.txt` | `abcabcabd` (no trailing newline; good for checking n-gram counts by hand) |
| `empty.txt` | empty |
| `crlf.txt` | Windows and old-Mac line endings |
| `bom.txt` | `hi` with a UTF-8 byte-order mark |

`data/sample.txt` is a slightly larger English sample for demos.
`.gitattributes` stops Git from changing the line endings in the fixtures.
