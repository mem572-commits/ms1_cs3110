type error =
  | File_not_found of string
  | Not_a_file of string
  | Read_error of string * string
  | Empty_text
  | Too_short of {
      length : int;
      required : int;
    }
  | Invalid_context_length of int
  | Invalid_ratio of float
  | Unknown_chars of char list

let error_to_string = function
  | File_not_found path -> Printf.sprintf "File not found: %s" path
  | Not_a_file path -> Printf.sprintf "Not a file (it is a directory): %s" path
  | Read_error (path, msg) -> Printf.sprintf "Could not read %s: %s" path msg
  | Empty_text -> "The training text is empty."
  | Too_short { length; required } ->
      Printf.sprintf
        "The text is too short: it has %d character(s) but at least %d are \
         needed."
        length required
  | Invalid_context_length n ->
      Printf.sprintf "Context length must be 0 or more, but got %d." n
  | Invalid_ratio r ->
      Printf.sprintf
        "Train ratio must be between 0 and 1 (exclusive), but got %g." r
  | Unknown_chars cs ->
      Printf.sprintf "These characters never appear in the training text: %s"
        (String.concat " " (List.map Vocab.display_char cs))

(* ---------- Reading text ---------- *)

let bom = "\xEF\xBB\xBF"

let strip_bom s =
  let n = String.length bom in
  if String.length s >= n && String.sub s 0 n = bom then
    String.sub s n (String.length s - n)
  else s

let normalize s =
  let s = strip_bom s in
  let n = String.length s in
  let buf = Buffer.create n in
  let rec go i =
    if i < n then
      match s.[i] with
      | '\r' ->
          Buffer.add_char buf '\n';
          if i + 1 < n && s.[i + 1] = '\n' then go (i + 2) else go (i + 1)
      | c ->
          Buffer.add_char buf c;
          go (i + 1)
  in
  go 0;
  Buffer.contents buf

let read_file path =
  if not (Sys.file_exists path) then Error (File_not_found path)
  else if Sys.is_directory path then Error (Not_a_file path)
  else
    match In_channel.with_open_bin path In_channel.input_all with
    | contents -> Ok (normalize contents)
    | exception Sys_error msg -> Error (Read_error (path, msg))

(* ---------- Corpora ---------- *)

(* AF: [{ text; vocab; ids }] is the training text [text], encoded with [vocab].

   RI: [text <> ""]. Every character of [text] is in [vocab]. [ids =
   Vocab.encode vocab text]. [ids] is never shared outside this module, so it
   cannot be mutated. *)
type t = {
  text : string;
  vocab : Vocab.t;
  ids : int array;
}

let of_string_with_vocab vocab text =
  if text = "" then Error Empty_text
  else
    match Vocab.try_encode vocab text with
    | Ok ids -> Ok { text; vocab; ids }
    | Error cs -> Error (Unknown_chars cs)

let of_string text = of_string_with_vocab (Vocab.of_string text) text
let load path = Result.bind (read_file path) of_string
let text c = c.text
let vocab c = c.vocab
let length c = String.length c.text
let ids c = Array.copy c.ids

let id_at c i =
  if i < 0 || i >= length c then
    invalid_arg
      (Printf.sprintf "Corpus.id_at: index %d is not in 0..%d" i (length c - 1))
  else c.ids.(i)

let encode_prompt c prompt =
  match Vocab.try_encode c.vocab prompt with
  | Ok ids -> Ok ids
  | Error cs -> Error (Unknown_chars cs)

let sub c start len =
  {
    text = String.sub c.text start len;
    vocab = c.vocab;
    ids = Array.sub c.ids start len;
  }

let split c ~train_ratio =
  if not (train_ratio > 0. && train_ratio < 1.) then
    Error (Invalid_ratio train_ratio)
  else
    let n = length c in
    let n_train = int_of_float (Float.of_int n *. train_ratio) in
    if n_train < 1 || n_train >= n then
      Error (Too_short { length = n; required = 2 })
    else Ok (sub c 0 n_train, sub c n_train (n - n_train))

let summary c =
  Printf.sprintf "Characters: %d\nVocabulary size: %d\nVocabulary: %s"
    (length c) (Vocab.size c.vocab) (Vocab.to_string c.vocab)

(* ---------- Training examples ---------- *)

type example = {
  context : int array;
  target : int;
}

let check_context c ~context =
  if context < 0 then Error (Invalid_context_length context)
  else if length c <= context then
    Error (Too_short { length = length c; required = context + 1 })
  else Ok (length c - context)

let num_examples c ~context = check_context c ~context

let examples_seq c ~context =
  Result.map
    (fun count ->
      Seq.init count (fun i ->
          { context = Array.sub c.ids i context; target = c.ids.(i + context) }))
    (check_context c ~context)

let examples c ~context = Result.map List.of_seq (examples_seq c ~context)
