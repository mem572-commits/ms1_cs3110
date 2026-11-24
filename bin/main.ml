(* Command-line interface for OcamlGPT. Loads a training file, fits an n-gram
   model on it, and prints generated text. *)

open Ocamlgpt

(* ---------- Settings ---------- *)

type config = {
  path : string;
  context : int;
  length : int;
  seed : int;
  prompt : string option;
}

let usage =
  "usage: ocamlgpt <training-file> [-k CONTEXT] [-n LENGTH] [-seed SEED] \
   [-prompt TEXT]"

(* [parse_args ()] is the config given on the command line. Exits with code 2
   if there is not exactly one training file. *)
let parse_args () =
  let path = ref "" in
  let context = ref 3 in
  let length = ref 200 in
  let seed = ref 3110 in
  let prompt = ref None in
  let specs =
    [
      ("-k", Arg.Set_int context, "CONTEXT  context length (default 3)");
      ("-n", Arg.Set_int length, "LENGTH  characters to generate (default 200)");
      ("-seed", Arg.Set_int seed, "SEED  random seed (default 3110)");
      ( "-prompt",
        Arg.String (fun s -> prompt := Some s),
        "TEXT  starting text (default: start of the training file)" );
    ]
  in
  let set_path s =
    if !path = "" then path := s
    else raise (Arg.Bad "only one training file can be given")
  in
  Arg.parse specs set_path usage;
  if !path = "" then begin
    prerr_endline "Error: no training file given";
    prerr_endline usage;
    exit 2
  end;
  {
    path = !path;
    context = !context;
    length = !length;
    seed = !seed;
    prompt = !prompt;
  }

(* ---------- Running ---------- *)

(* [generate_text corpus model config] is the prompt followed by
   [config.length] generated characters. With no prompt, the first
   [config.context] characters of the training text are used. The prompt and
   length are checked here since Ngram.generate raises Invalid_argument on
   them. *)
let generate_text corpus model config =
  let prompt =
    match config.prompt with
    | Some p -> p
    | None -> String.sub (Corpus.text corpus) 0 config.context
  in
  match Corpus.encode_prompt corpus prompt with
  | Error e -> Error (Corpus.error_to_string e)
  | Ok ids ->
      if Array.length ids < config.context then
        Error
          (Printf.sprintf "the prompt has %d characters but needs at least %d"
             (Array.length ids) config.context)
      else if config.length < 0 then
        Error
          (Printf.sprintf "the length must be 0 or more, but got %d"
             config.length)
      else
        let rng = Random.State.make [| config.seed |] in
        let out = Ngram.generate model rng ~prompt:ids ~length:config.length in
        let text = Vocab.decode (Corpus.vocab corpus) out in
        Ok (prompt ^ text)

(* [run config] is the text to print, or an error message. Nothing is printed
   here, so all output happens in one place below. *)
let run config =
  match Corpus.load config.path with
  | Error e -> Error (Corpus.error_to_string e)
  | Ok corpus -> (
      match Ngram.fit corpus ~context:config.context with
      | Error e -> Error (Corpus.error_to_string e)
      | Ok model -> (
          match generate_text corpus model config with
          | Error msg -> Error msg
          | Ok text ->
              let header =
                Printf.sprintf "n-gram model: context %d, vocabulary %d, seed %d"
                  (Ngram.context_length model) (Ngram.vocab_size model)
                  config.seed
              in
              Ok (header ^ "\n\n" ^ text)))

let () =
  let config = parse_args () in
  match run config with
  | Error msg ->
      prerr_endline ("Error: " ^ msg);
      exit 1
  | Ok output -> print_endline output
