open OUnit2
open Ocamlgpt

(* ---------- Helpers ---------- *)

let get_ok = function
  | Ok x -> x
  | Error e -> assert_failure ("unexpected error: " ^ Corpus.error_to_string e)

let assert_error expected = function
  | Ok _ -> assert_failure "expected an error, got Ok"
  | Error e -> assert_equal ~printer:Corpus.error_to_string expected e

let print_ints a =
  "[|" ^ String.concat "; " (Array.to_list (Array.map string_of_int a)) ^ "|]"

let print_chars cs = String.concat " " (List.map Vocab.display_char cs)
let fixture name = Filename.concat "fixtures" name
let corpus s = get_ok (Corpus.of_string s)

(* ---------- Vocab ---------- *)

let vocab_tests =
  let v = Vocab.of_string "hello" in
  [
    ( "size counts distinct chars" >:: fun _ ->
      assert_equal ~printer:string_of_int 4 (Vocab.size v) );
    ( "ids follow character order" >:: fun _ ->
      assert_equal ~printer:print_chars [ 'e'; 'h'; 'l'; 'o' ] (Vocab.chars v)
    );
    ( "empty vocabulary" >:: fun _ ->
      assert_equal 0 (Vocab.size (Vocab.of_string "")) );
    ( "order of input does not matter" >:: fun _ ->
      assert_bool "equal" (Vocab.equal v (Vocab.of_string "olleh")) );
    ( "of_chars ignores duplicates" >:: fun _ ->
      assert_bool "equal"
        (Vocab.equal v (Vocab.of_chars [ 'o'; 'h'; 'e'; 'l'; 'l' ])) );
    ( "id_of_char known" >:: fun _ ->
      assert_equal (Some 2) (Vocab.id_of_char v 'l') );
    ( "id_of_char unknown" >:: fun _ ->
      assert_equal None (Vocab.id_of_char v 'z') );
    ( "char_of_id in range" >:: fun _ ->
      assert_equal (Some 'o') (Vocab.char_of_id v 3) );
    ( "char_of_id out of range" >:: fun _ ->
      assert_equal None (Vocab.char_of_id v 4);
      assert_equal None (Vocab.char_of_id v (-1)) );
    ( "mem" >:: fun _ ->
      assert_bool "h in" (Vocab.mem v 'h');
      assert_bool "z out" (not (Vocab.mem v 'z')) );
    ( "encode" >:: fun _ ->
      assert_equal ~printer:print_ints [| 1; 0; 2; 2; 3 |]
        (Vocab.encode v "hello") );
    ("encode empty" >:: fun _ -> assert_equal [||] (Vocab.encode v ""));
    ( "encode unknown raises" >:: fun _ ->
      assert_raises (Vocab.Unknown_char 'z') (fun () -> Vocab.encode v "hez") );
    ( "decode" >:: fun _ ->
      assert_equal ~printer:Fun.id "hole" (Vocab.decode v [| 1; 3; 2; 0 |]) );
    ( "decode out of range raises" >:: fun _ ->
      assert_raises (Invalid_argument "Vocab.decode_id: id 9 is not in 0..3")
        (fun () -> Vocab.decode v [| 0; 9 |]) );
    ( "try_encode ok" >:: fun _ ->
      assert_equal (Ok [| 3; 1 |]) (Vocab.try_encode v "oh") );
    ( "try_encode lists unknown chars once, in order" >:: fun _ ->
      assert_equal (Error [ 'z'; '!' ]) (Vocab.try_encode v "hz!zz") );
    ( "unknown_chars none" >:: fun _ ->
      assert_equal [] (Vocab.unknown_chars v "hello") );
    ( "serialize round trip" >:: fun _ ->
      assert_equal ~printer:Fun.id "ehlo" (Vocab.serialize v);
      assert_bool "equal" (Vocab.equal v (Vocab.of_string (Vocab.serialize v)))
    );
    ( "display_char special chars" >:: fun _ ->
      assert_equal "\\n" (Vocab.display_char '\n');
      assert_equal "' '" (Vocab.display_char ' ');
      assert_equal "\\x00" (Vocab.display_char '\000');
      assert_equal "a" (Vocab.display_char 'a') );
    ( "to_string" >:: fun _ ->
      assert_equal ~printer:Fun.id "{\\n ' ' a}"
        (Vocab.to_string (Vocab.of_string "a a\n")) );
  ]

(* ---------- Reading files ---------- *)

let file_tests =
  [
    ( "normalize CRLF and CR" >:: fun _ ->
      assert_equal ~printer:String.escaped "a\nb\nc\n\n"
        (Corpus.normalize "a\r\nb\rc\n\r\n") );
    ( "normalize strips BOM" >:: fun _ ->
      assert_equal ~printer:String.escaped "hi"
        (Corpus.normalize "\xEF\xBB\xBFhi") );
    ( "normalize leaves plain text alone" >:: fun _ ->
      assert_equal "plain\ntext" (Corpus.normalize "plain\ntext") );
    ( "read_file" >:: fun _ ->
      assert_equal (Ok "hello\n") (Corpus.read_file (fixture "hello.txt")) );
    ( "read_file normalizes line endings" >:: fun _ ->
      assert_equal (Ok "one\ntwo\nthree\n")
        (Corpus.read_file (fixture "crlf.txt")) );
    ( "read_file strips BOM" >:: fun _ ->
      assert_equal (Ok "hi") (Corpus.read_file (fixture "bom.txt")) );
    ( "read_file empty file" >:: fun _ ->
      assert_equal (Ok "") (Corpus.read_file (fixture "empty.txt")) );
    ( "read_file missing" >:: fun _ ->
      assert_error (Corpus.File_not_found "fixtures/nope.txt")
        (Corpus.read_file "fixtures/nope.txt") );
    ( "read_file directory" >:: fun _ ->
      assert_error (Corpus.Not_a_file "fixtures") (Corpus.read_file "fixtures")
    );
    ( "load empty file" >:: fun _ ->
      assert_error Corpus.Empty_text (Corpus.load (fixture "empty.txt")) );
    ( "load builds vocabulary" >:: fun _ ->
      let c = get_ok (Corpus.load (fixture "hello.txt")) in
      assert_equal ~printer:print_chars
        [ '\n'; 'e'; 'h'; 'l'; 'o' ]
        (Vocab.chars (Corpus.vocab c)) );
  ]

(* ---------- Corpus ---------- *)

let abc = corpus "abcabcabd"

let corpus_tests =
  [
    ( "of_string empty" >:: fun _ ->
      assert_error Corpus.Empty_text (Corpus.of_string "") );
    ( "text and length" >:: fun _ ->
      assert_equal "abcabcabd" (Corpus.text abc);
      assert_equal 9 (Corpus.length abc) );
    ( "ids" >:: fun _ ->
      assert_equal ~printer:print_ints
        [| 0; 1; 2; 0; 1; 2; 0; 1; 3 |]
        (Corpus.ids abc) );
    ( "ids returns a copy" >:: fun _ ->
      let a = Corpus.ids abc in
      a.(0) <- 99;
      assert_equal 0 (Corpus.id_at abc 0) );
    ("id_at" >:: fun _ -> assert_equal 3 (Corpus.id_at abc 8));
    ( "id_at out of range" >:: fun _ ->
      assert_raises (Invalid_argument "Corpus.id_at: index 9 is not in 0..8")
        (fun () -> Corpus.id_at abc 9) );
    ( "of_string_with_vocab ok" >:: fun _ ->
      let c = get_ok (Corpus.of_string_with_vocab (Corpus.vocab abc) "dab") in
      assert_equal [| 3; 0; 1 |] (Corpus.ids c) );
    ( "of_string_with_vocab unknown" >:: fun _ ->
      assert_error (Corpus.Unknown_chars [ 'x' ])
        (Corpus.of_string_with_vocab (Corpus.vocab abc) "axa") );
    ( "encode_prompt" >:: fun _ ->
      assert_equal (Ok [| 2; 0 |]) (Corpus.encode_prompt abc "ca") );
    ( "encode_prompt empty" >:: fun _ ->
      assert_equal (Ok [||]) (Corpus.encode_prompt abc "") );
    ( "encode_prompt unknown" >:: fun _ ->
      assert_error (Corpus.Unknown_chars [ 'Q' ])
        (Corpus.encode_prompt abc "aQ") );
    ( "split" >:: fun _ ->
      let train, valid = get_ok (Corpus.split abc ~train_ratio:0.7) in
      assert_equal ~printer:Fun.id "abcabc" (Corpus.text train);
      assert_equal ~printer:Fun.id "abd" (Corpus.text valid) );
    ( "split keeps full vocabulary" >:: fun _ ->
      let train, valid = get_ok (Corpus.split abc ~train_ratio:0.5) in
      assert_bool "train" (Vocab.equal (Corpus.vocab abc) (Corpus.vocab train));
      assert_bool "valid" (Vocab.equal (Corpus.vocab abc) (Corpus.vocab valid));
      assert_equal [| 0; 1; 2; 0 |] (Corpus.ids train) );
    ( "split bad ratio" >:: fun _ ->
      assert_error (Corpus.Invalid_ratio 1.) (Corpus.split abc ~train_ratio:1.);
      assert_error (Corpus.Invalid_ratio 0.) (Corpus.split abc ~train_ratio:0.)
    );
    ( "split too short" >:: fun _ ->
      assert_error
        (Corpus.Too_short { length = 1; required = 2 })
        (Corpus.split (corpus "a") ~train_ratio:0.5);
      assert_error
        (Corpus.Too_short { length = 9; required = 2 })
        (Corpus.split abc ~train_ratio:0.05) );
    ( "summary" >:: fun _ ->
      assert_equal ~printer:Fun.id
        "Characters: 9\nVocabulary size: 4\nVocabulary: {a b c d}"
        (Corpus.summary abc) );
  ]

(* ---------- Examples ---------- *)

let ex context target = { Corpus.context; target }

let print_examples es =
  String.concat ", "
    (List.map
       (fun { Corpus.context; target } ->
         print_ints context ^ "->" ^ string_of_int target)
       es)

let example_tests =
  let abcd = corpus "abcd" in
  [
    ( "context 2" >:: fun _ ->
      assert_equal ~printer:print_examples
        [ ex [| 0; 1 |] 2; ex [| 1; 2 |] 3 ]
        (get_ok (Corpus.examples abcd ~context:2)) );
    ( "context 0 is one example per char" >:: fun _ ->
      assert_equal ~printer:print_examples
        [ ex [||] 0; ex [||] 1; ex [||] 2; ex [||] 3 ]
        (get_ok (Corpus.examples abcd ~context:0)) );
    ( "largest context" >:: fun _ ->
      assert_equal
        [ ex [| 0; 1; 2 |] 3 ]
        (get_ok (Corpus.examples abcd ~context:3)) );
    ( "context too long" >:: fun _ ->
      assert_error
        (Corpus.Too_short { length = 4; required = 5 })
        (Corpus.examples abcd ~context:4) );
    ( "negative context" >:: fun _ ->
      assert_error (Corpus.Invalid_context_length (-1))
        (Corpus.examples abcd ~context:(-1)) );
    ( "num_examples" >:: fun _ ->
      assert_equal (Ok 6) (Corpus.num_examples abc ~context:3) );
    ( "examples_seq can be traversed twice" >:: fun _ ->
      let s = get_ok (Corpus.examples_seq abc ~context:2) in
      assert_equal (List.of_seq s) (List.of_seq s) );
  ]

(* ---------- Properties ---------- *)

let nonempty_text = QCheck.(string_size_of Gen.(1 -- 60) Gen.printable)

let property_tests =
  List.map
    (QCheck_ounit.to_ounit2_test ~rand:(Random.State.make [| 3110 |]))
    [
      QCheck.Test.make ~name:"decode (encode s) = s" ~count:300 QCheck.string
        (fun s ->
          let v = Vocab.of_string s in
          Vocab.decode v (Vocab.encode v s) = s);
      QCheck.Test.make ~name:"every id is in range" ~count:300 QCheck.string
        (fun s ->
          let v = Vocab.of_string s in
          Array.for_all (fun i -> i >= 0 && i < Vocab.size v) (Vocab.encode v s));
      QCheck.Test.make ~name:"split parts join to the original" ~count:300
        QCheck.(pair nonempty_text (float_range 0.01 0.99))
        (fun (s, r) ->
          match Corpus.split (corpus s) ~train_ratio:r with
          | Ok (a, b) -> Corpus.text a ^ Corpus.text b = s
          | Error (Corpus.Too_short _) -> true
          | Error _ -> false);
      QCheck.Test.make ~name:"examples rebuild the text" ~count:300
        QCheck.(pair nonempty_text (int_bound 5))
        (fun (s, k) ->
          let c = corpus s in
          match Corpus.examples c ~context:k with
          | Error (Corpus.Too_short _) -> String.length s <= k
          | Error _ -> false
          | Ok es ->
              let ids = Corpus.ids c in
              List.length es = String.length s - k
              && List.for_all
                   (fun (i, { Corpus.context; target }) ->
                     context = Array.sub ids i k && target = ids.(i + k))
                   (List.mapi (fun i e -> (i, e)) es));
    ]

let () =
  run_test_tt_main
    ("ocamlgpt text processing"
    >::: [
           "vocab" >::: vocab_tests;
           "files" >::: file_tests;
           "corpus" >::: corpus_tests;
           "examples" >::: example_tests;
           "properties" >::: property_tests;
         ])
