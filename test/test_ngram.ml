open OUnit2
open Ocamlgpt

(* ---------- Helpers ---------- *)

let get_ok = function
  | Ok x -> x
  | Error e -> assert_failure ("unexpected error: " ^ Corpus.error_to_string e)

let corpus s = get_ok (Corpus.of_string s)

(* "abcabd": ids are a=0, b=1, c=2, d=3 (Vocab sorts characters) *)
let c = corpus "abcabd"
let m2 = get_ok (Ngram.fit c ~context:2)


(* ---------- Counts ---------- *)

let count_tests =
  [
    ( "ab followed by c once" >:: fun _ ->
      assert_equal ~printer:string_of_int 1 (Ngram.count m2 [| 0; 1 |] 2) );
    ( "ab followed by d once" >:: fun _ -> 
        assert_equal ~printer:string_of_int 1 (Ngram.count m2 [| 0; 1 |] 3) );
    ( "ab never followed by a" >:: fun _ -> 
        assert_equal ~printer:string_of_int 0 (Ngram.count m2 [| 0; 1 |] 0) );
    ( "unseen context bd gets value 0" >:: fun _ -> 
        assert_equal ~printer:string_of_int 0 (Ngram.count m2 [| 1; 3 |] 2) );

  ]

(* ---------- Probabilities ---------- *)

let predict_tests = 
    [ 
        ( "sums to one for a seen context" >:: fun _ ->
            let p = Ngram.predict m2 [| 0; 1 |] in
            let sum = Array.fold_left ( +. ) 0. p in
            assert_bool "sums to 1" (Float.abs (sum -. 1.) < 1e-9) );
        ( "uniform prediction for unseen context" >:: fun _ ->
            let p = Ngram.predict m2 [| 1; 3 |] in
            assert_bool "uniform" (Array.for_all (fun x -> x = 0.25) p) );
    ]

(* ---------- Generation ---------- *)

let generate_tests = [ (* output length; same seed gives same output *) ]

(* ---------- Errors ---------- *)

let error_tests = [ (* fit with negative context; short prompt; negative length *) ]

let () =
  run_test_tt_main
    ("ngram"
    >::: [
           "counts" >::: count_tests;
           "predict" >::: predict_tests;
           "generate" >::: generate_tests;
           "errors" >::: error_tests;
         ])