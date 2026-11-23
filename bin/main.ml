(* Placeholder entry point that shows the text-processing layer working end to
   end. The real command-line interface (model choice, generation settings) will
   replace this; see docs/TEXT_PROCESSING.md for the API it builds on. *)

open Ocamlgpt

let usage = "usage: ocamlgpt <training-file> [context-length]"

let run path context =
  match Corpus.load path with
  | Error e ->
      prerr_endline ("Error: " ^ Corpus.error_to_string e);
      exit 1
  | Ok corpus -> (
      print_endline (Corpus.summary corpus);
      match Corpus.num_examples corpus ~context with
      | Error e ->
          prerr_endline ("Error: " ^ Corpus.error_to_string e);
          exit 1
      | Ok n -> Printf.printf "Training examples (context %d): %d\n" context n)

let () =
  match Array.to_list Sys.argv with
  | [ _; path ] -> run path 3
  | [ _; path; n ] -> (
      match int_of_string_opt n with
      | Some context -> run path context
      | None ->
          prerr_endline
            ("Error: context length must be a whole number, got " ^ n);
          exit 2)
  | _ ->
      prerr_endline usage;
      exit 2
