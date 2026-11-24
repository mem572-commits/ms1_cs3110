open OUnit2

(* End-to-end tests for the ocamlgpt command. Each test runs the real program,
   so file loading, the vocabulary, Ngram.fit, Ngram.generate, and decoding are
   all checked together. *)

(* ---------- Helpers ---------- *)

(* [run args] runs ocamlgpt with [args] and is its exit code, standard output,
   and standard error. The output goes through files since Sys.command only
   gives back the exit code. Each run gets new files because OUnit can run
   tests at the same time. *)
let run args =
  let out_file = Filename.temp_file "cli" ".out" in
  let err_file = Filename.temp_file "cli" ".err" in
  let code =
    Sys.command
      ("../bin/main.exe " ^ args ^ " > " ^ out_file ^ " 2> " ^ err_file)
  in
  let read file = In_channel.with_open_text file In_channel.input_all in
  let out = read out_file in
  let err = read err_file in
  Sys.remove out_file;
  Sys.remove err_file;
  (code, out, err)

(* [text_of out] is the generated text in [out], without the header line, the
   blank line after it, and the final newline. *)
let text_of out =
  let i = String.index out '\n' in
  String.sub out (i + 2) (String.length out - i - 3)

(* [generated args] is the text printed by a run that should succeed. *)
let generated args =
  let code, out, err = run args in
  assert_equal ~printer:string_of_int ~msg:err 0 code;
  text_of out

(* [assert_fails args code msg] checks that a run exits with [code] and prints
   [msg] as its first line of standard error. *)
let assert_fails args code msg =
  let c, out, err = run args in
  assert_equal ~printer:string_of_int code c;
  assert_equal ~printer:Fun.id "" out;
  assert_equal ~printer:Fun.id msg (List.hd (String.split_on_char '\n' err))

let forced = "fixtures/forced.txt"
let sample = "../data/sample.txt"

(* ---------- Forced output ---------- *)

(* "abcabcabc" with context 2: ab is only ever followed by c, bc by a, and ca
   by b. Every step is forced, so the output is the same for any seed and can
   be worked out by hand. *)
let forced_tests =
  [
    ( "header shows the settings" >:: fun _ ->
      let _, out, _ = run (forced ^ " -k 2 -n 7 -prompt ab -seed 0") in
      assert_equal ~printer:Fun.id
        "n-gram model: context 2, vocabulary 3, seed 0\n\nabcabcabc\n" out );
    ( "same output for seed 0 and seed 3110" >:: fun _ ->
      assert_equal ~printer:Fun.id "abcabcabc"
        (generated (forced ^ " -k 2 -n 7 -prompt ab -seed 0"));
      assert_equal ~printer:Fun.id "abcabcabc"
        (generated (forced ^ " -k 2 -n 7 -prompt ab -seed 3110")) );
    ( "no prompt uses the start of the file" >:: fun _ ->
      assert_equal ~printer:Fun.id "abcabc" (generated (forced ^ " -k 2 -n 4"))
    );
    ( "a long prompt only uses its last 2 characters" >:: fun _ ->
      (* ...ca is followed by b, then bc by a, then ab by c *)
      assert_equal ~printer:Fun.id "abcabca"
        (generated (forced ^ " -k 2 -n 3 -prompt abca")) );
  ]

(* ---------- Seeded runs on data/sample.txt ---------- *)

(* This output was recorded from the program, not worked out by hand. It
   checks that the same seed always gives the same text. It depends on OCaml
   5's Random, so it would change on OCaml 4. *)
let seed_3110_output =
  "the lazy dog from the dog fox runs the quick brown fox and the quick fox \
   runs to the fox runs the dog fox and the fox jumps in the dog.\n\
   the fox jumps in the fox and the old stone wall cat wall.\n\
   fbpcqvtuk"

let sample_tests =
  [
    ( "seed 3110 gives the recorded output" >:: fun _ ->
      assert_equal ~printer:Fun.id seed_3110_output
        (generated (sample ^ " -k 3 -n 200 -seed 3110 -prompt the")) );
    ( "output is the prompt plus the requested length" >:: fun _ ->
      assert_equal ~printer:string_of_int 203
        (String.length (generated (sample ^ " -k 3 -n 200 -prompt the"))) );
    ( "running twice with the same seed gives the same text" >:: fun _ ->
      let args = sample ^ " -k 3 -n 150 -seed 42 -prompt the" in
      assert_equal ~printer:Fun.id (generated args) (generated args) );
    ( "a different seed gives different text" >:: fun _ ->
      assert_bool "same text"
        (generated (sample ^ " -n 100 -seed 7 -prompt the")
        <> generated (sample ^ " -n 100 -seed 3110 -prompt the")) );
    ( "context 0 works without a prompt" >:: fun _ ->
      assert_equal ~printer:string_of_int 40
        (String.length (generated (sample ^ " -k 0 -n 40"))) );
    ( "length 0 prints only the prompt" >:: fun _ ->
      assert_equal ~printer:Fun.id "the" (generated (sample ^ " -n 0 -prompt the"))
    );
  ]

(* ---------- Errors ---------- *)

let error_tests =
  [
    ( "prompt shorter than the context" >:: fun _ ->
      assert_fails (sample ^ " -prompt th") 1
        "Error: the prompt has 2 characters but needs at least 3" );
    ( "prompt with a character not in the training text" >:: fun _ ->
      assert_fails (sample ^ " -prompt theZ") 1
        "Error: These characters never appear in the training text: Z" );
    ( "negative length" >:: fun _ ->
      assert_fails (sample ^ " -n -1") 1
        "Error: the length must be 0 or more, but got -1" );
    ( "negative context" >:: fun _ ->
      assert_fails (sample ^ " -k -1") 1
        "Error: Context length must be 0 or more, but got -1." );
    ( "context as long as the file" >:: fun _ ->
      assert_fails (forced ^ " -k 9") 1
        "Error: The text is too short: it has 9 character(s) but at least 10 \
         are needed." );
    ( "missing file" >:: fun _ ->
      assert_fails "missing.txt" 1 "Error: File not found: missing.txt" );
    ( "no training file" >:: fun _ ->
      assert_fails "" 2 "Error: no training file given" );
  ]

let () =
  run_test_tt_main
    ("cli"
    >::: [
           "forced" >::: forced_tests;
           "sample" >::: sample_tests;
           "errors" >::: error_tests;
         ])
