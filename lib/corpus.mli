(** Training text and datasets.

    A corpus is a piece of training text together with its {!Vocab.t} and the
    text already encoded as ids. This is the starting point for every model:

    {[
      match Corpus.load "data/sample.txt" with
      | Error e -> prerr_endline (Corpus.error_to_string e)
      | Ok corpus ->
          let vocab = Corpus.vocab corpus in
          (* n-gram model: count (context, next character) pairs *)
          Corpus.examples_seq corpus ~context:3
          |> Seq.iter (fun { Corpus.context; target } -> ...)
    ]}

    Functions that can fail because of user input return a [result] with an
    {!error}, so the command-line interface can print a clear message with
    {!error_to_string} instead of crashing. *)

(** {1 Errors} *)

type error =
  | File_not_found of string  (** No file exists at this path. *)
  | Not_a_file of string  (** The path names a directory, not a file. *)
  | Read_error of string * string
      (** [Read_error (path, msg)]: the file exists but could not be read. *)
  | Empty_text  (** The text has no characters. *)
  | Too_short of {
      length : int;
      required : int;
    }
      (** The text has [length] characters but at least [required] are needed.
      *)
  | Invalid_context_length of int  (** A context length was negative. *)
  | Invalid_ratio of float
      (** A train/validation ratio was not strictly between 0 and 1. *)
  | Unknown_chars of char list
      (** The text uses these characters, which are not in the vocabulary. *)

val error_to_string : error -> string
(** [error_to_string e] is a one-line message describing [e], suitable for
    showing to the user. *)

(** {1 Reading text} *)

val normalize : string -> string
(** [normalize s] is [s] with a leading UTF-8 byte-order mark removed and every
    Windows ["\r\n"] or old-Mac ["\r"] line ending replaced by ["\n"]. This
    makes a file give the same characters on every operating system. *)

val read_file : string -> (string, error) result
(** [read_file path] is [Ok text], where [text] is the normalized contents of
    the file at [path]. It is [Error] if the file is missing, is a directory, or
    cannot be read. An empty file gives [Ok ""]. *)

(** {1 Corpora} *)

type t
(** A non-empty text, its vocabulary, and its encoding. Immutable. *)

val of_string : string -> (t, error) result
(** [of_string text] is a corpus of [text] whose vocabulary is the characters of
    [text]. [text] is used exactly as given (it is not normalized). Returns
    [Error Empty_text] if [text = ""]. *)

val of_string_with_vocab : Vocab.t -> string -> (t, error) result
(** [of_string_with_vocab v text] is a corpus of [text] that uses the existing
    vocabulary [v]. Use this for held-out or evaluation text so that its ids
    match the model's ids. Returns [Error Empty_text] if [text = ""], or
    [Error (Unknown_chars cs)] if [text] uses characters not in [v]. *)

val load : string -> (t, error) result
(** [load path] reads and normalizes the file at [path] (see {!read_file}) and
    builds a corpus from it (see {!of_string}). *)

val text : t -> string
(** [text c] is the text of [c]. *)

val vocab : t -> Vocab.t
(** [vocab c] is the vocabulary of [c]. *)

val length : t -> int
(** [length c] is the number of characters in [c]. Always at least 1. *)

val ids : t -> int array
(** [ids c] is the encoding of [text c]: [Vocab.encode (vocab c) (text c)]. The
    result is a fresh copy, so changing it does not change [c]. *)

val id_at : t -> int -> int
(** [id_at c i] is the id of the [i]-th character of [c] (counting from 0).
    Raises [Invalid_argument] if [i] is not in [0 .. length c - 1]. *)

val encode_prompt : t -> string -> (int array, error) result
(** [encode_prompt c prompt] is the encoding of [prompt] using [vocab c]. The
    empty prompt gives [Ok [||]]. Returns [Error (Unknown_chars cs)] if [prompt]
    uses characters that never appear in the training text. *)

val split : t -> train_ratio:float -> (t * t, error) result
(** [split c ~train_ratio] is [Ok (train, valid)], where [train] holds the first
    [floor (train_ratio *. float (length c))] characters of [c] and [valid]
    holds the rest. Both parts keep the vocabulary of [c], even if some
    characters appear in only one part, so ids agree across them. Returns
    [Error (Invalid_ratio r)] if the ratio is not strictly between 0 and 1, or
    [Error (Too_short _)] if either part would be empty. *)

val summary : t -> string
(** [summary c] is a short multi-line report of the number of characters, the
    vocabulary size, and the vocabulary itself. *)

(** {1 Training examples}

    A training example pairs a fixed-length window of ids with the id that comes
    right after it. For the text ["abcd"] and [~context:2], the examples are
    [ab -> c] and [bc -> d] (shown as characters; the actual values are ids).
    [~context:0] gives one example per character with an empty context, which is
    what a unigram model needs.

    There is no padding: the first example starts at the start of the text, so a
    text of length [n] gives [n - context] examples. *)

type example = {
  context : int array;  (** The [context] ids before the target, in order. *)
  target : int;  (** The id that follows the context. *)
}

val num_examples : t -> context:int -> (int, error) result
(** [num_examples c ~context] is the number of examples that {!examples} gives,
    or the same error {!examples} would give. *)

val examples_seq : t -> context:int -> (example Seq.t, error) result
(** [examples_seq c ~context] is all the training examples of [c] with windows
    of length [context], in text order. The sequence is built lazily, which
    saves memory on large files, and can be traversed more than once. Returns
    [Error (Invalid_context_length context)] if [context < 0], or
    [Error (Too_short _)] if [length c <= context]. *)

val examples : t -> context:int -> (example list, error) result
(** [examples c ~context] is the same as {!examples_seq} but as a list. *)
