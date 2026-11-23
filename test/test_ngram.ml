open OUnit2
open Ocamlgpt

(* ---------- Helpers ---------- *)

let get_ok = function
  | Ok x -> x
  | Error e -> assert_failure ("unexpected error: " ^ Corpus.error_to_string e)

let print_ints a =
  "[|" ^ String.concat "; " (Array.to_list (Array.map string_of_int a)) ^ "|]"

let print_floats a =
  "[|"
  ^ String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%.6f") a))
  ^ "|]"

let floats_close a b =
  Array.length a = Array.length b
  && Array.for_all2 (fun x y -> Float.abs (x -. y) < 1e-9) a b

let raises_invalid_arg f =
  match f () with
  | _ -> assert_failure "expected Invalid_argument"
  | exception Invalid_argument _ -> ()

let corpus s = get_ok (Corpus.of_string s)

(* "abcabd": ids are a=0, b=1, c=2, d=3 (Vocab sorts characters) *)
let c = corpus "abcabd"
let m2 = get_ok (Ngram.fit c ~context:2)

(* Context 0: every example shares the empty context, so predict gives the
   unigram frequencies. In "abcabd": a=2, b=2, c=1, d=1 out of 6. *)
let m0 = get_ok (Ngram.fit c ~context:0)

(* "ababab": a=0, b=1. With context 1, a is always followed by b and b by a,
   so every prediction is certain. *)
let m1 = get_ok (Ngram.fit (corpus "ababab") ~context:1)

(* ---------- Accessors ---------- *)

let accessor_tests =
  [
    ( "vocab_size is the number of distinct chars" >:: fun _ ->
      assert_equal ~printer:string_of_int 4 (Ngram.vocab_size m2) );
    ( "context_length is the fitted context" >:: fun _ ->
      assert_equal ~printer:string_of_int 2 (Ngram.context_length m2) );
    ( "context_length 0 is allowed" >:: fun _ ->
      assert_equal ~printer:string_of_int 0 (Ngram.context_length m0) );
  ]

(* ---------- Fit ---------- *)

let fit_tests =
  [
    ( "fit succeeds when length = context + 1" >:: fun _ ->
      assert_bool "Ok" (Result.is_ok (Ngram.fit c ~context:5)) );
    ( "fit returns Too_short when length = context" >:: fun _ ->
      assert_equal
        (Error (Corpus.Too_short { length = 6; required = 7 }))
        (Ngram.fit c ~context:6) );
    ( "fit returns Too_short when context exceeds length" >:: fun _ ->
      assert_equal
        (Error (Corpus.Too_short { length = 6; required = 11 }))
        (Ngram.fit c ~context:10) );
  ]

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
    ( "context 0 counts every occurrence" >:: fun _ ->
      assert_equal ~printer:string_of_int 2 (Ngram.count m0 [||] 0) );
  ]

(* ---------- Probabilities ---------- *)

let predict_tests =
  [
    ( "sums to one for a seen context" >:: fun _ ->
      let p = Ngram.predict m2 [| 0; 1 |] in
      let sum = Array.fold_left ( +. ) 0. p in
      assert_bool "sums to 1" (Float.abs (sum -. 1.) < 1e-9) );
    ( "seen context gives count frequencies" >:: fun _ ->
        assert_equal ~printer:print_floats ~cmp:floats_close
        [| 0.; 0.; 0.5; 0.5 |] 
        (Ngram.predict m2 [| 0; 1 |]) );
    ( "uniform prediction for unseen context" >:: fun _ ->
      let p = Ngram.predict m2 [| 1; 3 |] in
      assert_bool "uniform" (Array.for_all (fun x -> x = 0.25) p) );
    ( "context 0 gives unigram frequencies" >:: fun _ ->
      assert_equal ~printer:print_floats ~cmp:floats_close
        [| 1. /. 3.; 1. /. 3.; 1. /. 6.; 1. /. 6. |]
        (Ngram.predict m0 [||]) );
  ]

(* ---------- Sampling ---------- *)

let sample_next_tests =
  let all_seeds_give expected ctx =
    List.iter
      (fun seed ->
        assert_equal ~printer:string_of_int expected
          (Ngram.sample_next m1 (Random.State.make [| seed |]) ctx))
      [ 0; 1; 42; 3110; 99999 ]
  in
  [
    ( "a is always followed by b" >:: fun _ -> all_seeds_give 1 [| 0 |] );
    ( "b is always followed by a" >:: fun _ -> all_seeds_give 0 [| 1 |] );
    ( "sample_next result is a valid id" >:: fun _ ->
      let st = Random.State.make [| 3110 |] in
      for _ = 1 to 100 do
        let id = Ngram.sample_next m2 st [| 1; 3 |] in
        assert_bool "in range" (id >= 0 && id < Ngram.vocab_size m2)
      done );
    ( "sample_next only returns ids that followed the context" >:: fun _ ->
        let st = Random.State.make [| 3110 |] in
        let seen = Array.make 4 0 in
        for _ = 1 to 1000 do
            let id = Ngram.sample_next m2 st [| 0; 1 |] in
            seen.(id) <- seen.(id) + 1
        done;
        assert_equal ~printer:print_ints [| 0; 0 |] (Array.sub seen 0 2);
        assert_bool "c and d both roughly half" (seen.(2) > 400 && seen.(3) > 400) );
  ]

(* ---------- Generation ---------- *)

let generate_tests =
  let gen seed n =
    Ngram.generate m2 (Random.State.make [| seed |]) ~prompt:[| 0; 1 |] ~length:n
  in
  [
    ( "generate returns the requested length" >:: fun _ ->
      assert_equal ~printer:string_of_int 200 (Array.length (gen 3110 200)) );
    ( "length 0 gives empty array" >:: fun _ ->
      assert_equal ~printer:print_ints [||] (gen 3110 0) );
    ( "using the same seed twice gives the same output" >:: fun _ ->
      assert_equal ~printer:print_ints (gen 3110 200) (gen 3110 200) );
    ( "generated ids are all valid" >:: fun _ ->
      assert_bool "in range"
        (Array.for_all (fun id -> id >= 0 && id < Ngram.vocab_size m2) (gen 3110 200)) );
    ( "generate uses only the last context ids of a long prompt" >:: fun _ ->
      (* Only the final a (0) should matter, so output starts with b. *)
      assert_equal ~printer:print_ints [| 1; 0; 1; 0 |]
        (Ngram.generate m1 (Random.State.make [| 3110 |])
           ~prompt:[| 1; 1; 0 |] ~length:4) );
    ( "context 0 accepts an empty prompt" >:: fun _ ->
      assert_equal ~printer:string_of_int 10
        (Array.length
           (Ngram.generate m0 (Random.State.make [| 3110 |]) ~prompt:[||] ~length:10)) );
  ]

(* ---------- Properties ---------- *)
let nonempty_text = QCheck.(string_size_of Gen.(1 -- 60) Gen.printable)

let with_model (s, k) f =
  let c = corpus s in
  match Ngram.fit c ~context:k with
  | Error (Corpus.Too_short _) -> true
  | Error _ -> false
  | Ok m -> f c k m

let property_tests =
  List.map
    (QCheck_ounit.to_ounit2_test ~rand:(Random.State.make [| 3110 |]))
    [
      QCheck.Test.make ~name:"predict sums to 1 on every seen context" ~count:300
        QCheck.(pair nonempty_text (int_bound 4))
        (fun input ->
          with_model input (fun c k m ->
              List.for_all
                (fun { Corpus.context; _ } ->
                  let sum = Array.fold_left ( +. ) 0. (Ngram.predict m context) in
                  Float.abs (sum -. 1.) < 1e-9)
                (get_ok (Corpus.examples c ~context:k))));
      QCheck.Test.make ~name:"counts add up to the number of examples" ~count:300
        QCheck.(pair nonempty_text (int_bound 4))
        (fun input ->
          with_model input (fun c k m ->
              let ctxs =
                get_ok (Corpus.examples c ~context:k)
                |> List.map (fun e -> e.Corpus.context)
                |> List.sort_uniq compare
              in
              let total =
                List.fold_left
                  (fun acc ctx ->
                    let row = ref 0 in
                    for id = 0 to Ngram.vocab_size m - 1 do
                      row := !row + Ngram.count m ctx id
                    done;
                    acc + !row)
                  0 ctxs
              in
              total = Corpus.length c - k));
      QCheck.Test.make ~name:"generate gives in-range ids of the right length" ~count:300
        QCheck.(triple nonempty_text (int_bound 4) (int_bound 50))
        (fun (s, k, n) ->
          with_model (s, k) (fun c k m ->
              let prompt = Array.sub (Corpus.ids c) 0 k in
              let out = Ngram.generate m (Random.State.make [| 3110 |]) ~prompt ~length:n in
              Array.length out = n
              && Array.for_all (fun id -> id >= 0 && id < Ngram.vocab_size m) out));
    ]

(* ---------- Errors ---------- *)

let error_tests =
  [
    ( "fit rejects negative context" >:: fun _ ->
      assert_equal (Error (Corpus.Invalid_context_length (-1))) (Ngram.fit c ~context:(-1)) );
    ( "count raises invalid arg for a context that is too short" >:: fun _ ->
      raises_invalid_arg (fun () -> Ngram.count m2 [| 0 |] 1) );
    ( "count raises invalid arg for a context that is too long" >:: fun _ ->
      raises_invalid_arg (fun () -> Ngram.count m2 [| 0; 1; 2 |] 1) );
    ( "count raises invalid arg for id = -1" >:: fun _ ->
      raises_invalid_arg (fun () -> Ngram.count m2 [| 0; 1 |] (-1)) );
    ( "count raises invalid arg for id = vocab_size" >:: fun _ ->
      raises_invalid_arg (fun () -> Ngram.count m2 [| 0; 1 |] (Ngram.vocab_size m2)) );
    ( "predict raises invalid arg for a context that is too short" >:: fun _ ->
      raises_invalid_arg (fun () -> Ngram.predict m2 [| 0 |]) );
    ( "predict raises invalid arg for a context that is too long" >:: fun _ ->
      raises_invalid_arg (fun () -> Ngram.predict m2 [| 0; 1; 2 |]) );
    ( "generate raises invalid arg if prompt is shorter than the context" >:: fun _ ->
      raises_invalid_arg (fun () ->
          Ngram.generate m2 (Random.State.make [| 3110 |]) ~prompt:[| 0 |] ~length:10) );
    ( "generate raises invalid arg if length is negative" >:: fun _ ->
      raises_invalid_arg (fun () ->
          Ngram.generate m2 (Random.State.make [| 3110 |]) ~prompt:[| 0; 1 |] ~length:(-1)) );
  ]

let () =
  run_test_tt_main
    ("ngram"
    >::: [
           "accessors" >::: accessor_tests;
           "fit" >::: fit_tests;
           "counts" >::: count_tests;
           "predict" >::: predict_tests;
           "sample_next" >::: sample_next_tests;
           "generate" >::: generate_tests;
           "properties" >::: property_tests;
           "errors" >::: error_tests;
         ])
