open OUnit2
open Ocamlgpt

(* Expected values in these tests were worked out by hand, not copied from
   the implementation. *)

(* ---------- Helpers ---------- *)

let print_vec v =
  "[|" ^ String.concat "; " (Array.to_list (Array.map string_of_float v)) ^ "|]"

let print_mat m = String.concat "\n" (Array.to_list (Array.map print_vec m))
let close ?(eps = 1e-9) a b = Float.abs (a -. b) <= eps

let vec_close ?eps a b =
  Array.length a = Array.length b
  && Array.for_all2 (fun x y -> close ?eps x y) a b

let assert_vec ?eps expected actual =
  assert_equal ~printer:print_vec ~cmp:(vec_close ?eps) expected actual

let raises_invalid f =
  match f () with
  | _ -> assert_failure "expected Invalid_argument"
  | exception Invalid_argument _ -> ()

(* ---------- Matrix ---------- *)

let m23 = [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6. |] |]

let matrix_tests =
  [
    ( "zeros has the right shape" >:: fun _ ->
      let z = Matrix.zeros 2 3 in
      assert_equal 2 (Matrix.rows z);
      assert_equal 3 (Matrix.cols z);
      assert_bool "all zero" (Array.for_all (Array.for_all (fun x -> x = 0.)) z) );
    ( "zeros negative size raises" >:: fun _ ->
      raises_invalid (fun () -> Matrix.zeros (-1) 2) );
    ( "cols of empty matrix" >:: fun _ -> assert_equal 0 (Matrix.cols [||]) );
    ( "random stays in range and is reproducible" >:: fun _ ->
      let a = Matrix.random (Random.State.make [| 7 |]) 4 5 ~scale:0.5 in
      let b = Matrix.random (Random.State.make [| 7 |]) 4 5 ~scale:0.5 in
      assert_equal ~printer:print_mat a b;
      assert_bool "in range"
        (Array.for_all (Array.for_all (fun x -> x >= -0.5 && x <= 0.5)) a) );
    ( "copy does not share rows" >:: fun _ ->
      let c = Matrix.copy m23 in
      c.(0).(0) <- 99.;
      assert_equal 1. m23.(0).(0) );
    ( "one_hot" >:: fun _ -> assert_vec [| 0.; 0.; 1.; 0. |] (Matrix.one_hot 4 2) );
    ( "one_hot out of range raises" >:: fun _ ->
      raises_invalid (fun () -> Matrix.one_hot 3 3) );
    (* 1*4 + 2*5 + 3*6 = 32 *)
    ( "dot" >:: fun _ ->
      assert_equal ~printer:string_of_float 32. (Matrix.dot [| 1.; 2.; 3. |] [| 4.; 5.; 6. |])
    );
    ( "dot length mismatch raises" >:: fun _ ->
      raises_invalid (fun () -> Matrix.dot [| 1. |] [| 1.; 2. |]) );
    ( "add_vec" >:: fun _ ->
      assert_vec [| 5.; 7. |] (Matrix.add_vec [| 1.; 2. |] [| 4.; 5. |]) );
    (* [1 2 3; 4 5 6] * [1; 0; -1] = [1 - 3; 4 - 6] = [-2; -2] *)
    ( "mat_vec 2x3" >:: fun _ ->
      assert_vec [| -2.; -2. |] (Matrix.mat_vec m23 [| 1.; 0.; -1. |]) );
    ( "mat_vec size mismatch raises" >:: fun _ ->
      raises_invalid (fun () -> Matrix.mat_vec m23 [| 1.; 2. |]) );
    (* [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50] *)
    ( "mat_mul 2x2" >:: fun _ ->
      assert_equal ~printer:print_mat
        [| [| 19.; 22. |]; [| 43.; 50. |] |]
        (Matrix.mat_mul [| [| 1.; 2. |]; [| 3.; 4. |] |] [| [| 5.; 6. |]; [| 7.; 8. |] |])
    );
    (* (2x3) * (3x1) gives 2x1, the same numbers as mat_vec above *)
    ( "mat_mul agrees with mat_vec" >:: fun _ ->
      assert_equal ~printer:print_mat
        [| [| -2. |]; [| -2. |] |]
        (Matrix.mat_mul m23 [| [| 1. |]; [| 0. |]; [| -1. |] |]) );
    ( "mat_mul size mismatch raises" >:: fun _ ->
      raises_invalid (fun () -> Matrix.mat_mul m23 m23) );
    ( "softmax of equal values is uniform" >:: fun _ ->
      assert_vec [| 0.5; 0.5 |] (Matrix.softmax [| 0.; 0. |]) );
    (* exp 0 = 1 and exp (ln 3) = 3, so the result is [1/4; 3/4] *)
    ( "softmax [0; ln 3]" >:: fun _ ->
      assert_vec [| 0.25; 0.75 |] (Matrix.softmax [| 0.; log 3. |]) );
    ( "softmax of huge values does not overflow" >:: fun _ ->
      assert_vec [| 0.5; 0.5 |] (Matrix.softmax [| 1000.; 1000. |]) );
    ( "softmax empty raises" >:: fun _ -> raises_invalid (fun () -> Matrix.softmax [||]) );
    ( "argmax" >:: fun _ -> assert_equal 2 (Matrix.argmax [| 1.; 3.; 5.; 2. |]) );
    ( "argmax tie picks first" >:: fun _ ->
      assert_equal 1 (Matrix.argmax [| 0.; 4.; 4. |]) );
    ( "argmax empty raises" >:: fun _ -> raises_invalid (fun () -> Matrix.argmax [||]) );
  ]

let matrix_properties =
  List.map
    (QCheck_ounit.to_ounit2_test ~rand:(Random.State.make [| 3110 |]))
    [
      QCheck.Test.make ~name:"softmax is positive and sums to 1" ~count:300
        QCheck.(array_size Gen.(int_range 1 20) (float_range (-50.) 50.))
        (fun v ->
          let p = Matrix.softmax v in
          Array.for_all (fun x -> x > 0.) p
          && close ~eps:1e-9 1. (Array.fold_left ( +. ) 0. p));
    ]

(* ---------- Neural: shape and input ---------- *)

let ex context target = { Corpus.context; target }

let get_ok = function
  | Ok x -> x
  | Error e -> assert_failure ("unexpected error: " ^ Corpus.error_to_string e)

let shape_tests =
  [
    ( "create has the requested sizes" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:3 ~seed:1 in
      assert_equal 4 (Neural.vocab_size m);
      assert_equal 3 (Neural.context_length m);
      assert_equal 4 (Matrix.rows (Neural.weights m));
      assert_equal 12 (Matrix.cols (Neural.weights m));
      assert_vec [| 0.; 0.; 0.; 0. |] (Neural.bias m) );
    ( "create with a bad size raises" >:: fun _ ->
      raises_invalid (fun () -> Neural.create ~vocab_size:0 ~context:1 ~seed:1);
      raises_invalid (fun () -> Neural.create ~vocab_size:3 ~context:(-1) ~seed:1) );
    ( "same seed gives the same model" >:: fun _ ->
      let a = Neural.create ~vocab_size:3 ~context:2 ~seed:5 in
      let b = Neural.create ~vocab_size:3 ~context:2 ~seed:5 in
      assert_equal ~printer:print_mat (Neural.weights a) (Neural.weights b) );
    ( "of_params with wrong weight shape raises" >:: fun _ ->
      raises_invalid (fun () ->
          Neural.of_params ~context:1 ~weights:(Matrix.zeros 2 3) ~bias:[| 0.; 0. |]) );
    ( "of_params copies its arrays" >:: fun _ ->
      let w = Matrix.zeros 2 2 in
      let m = Neural.of_params ~context:1 ~weights:w ~bias:[| 0.; 0. |] in
      w.(0).(0) <- 5.;
      assert_equal 0. (Neural.weights m).(0).(0) );
    ( "weights returns a copy" >:: fun _ ->
      let m = Neural.create ~vocab_size:2 ~context:1 ~seed:1 in
      let w = Neural.weights m in
      w.(0).(0) <- 5.;
      assert_bool "unchanged" ((Neural.weights m).(0).(0) <> 5.) );
    (* vocab 3: id 2 -> [0 0 1], id 0 -> [1 0 0] *)
    ( "input_of_context joins one-hot vectors" >:: fun _ ->
      let m = Neural.create ~vocab_size:3 ~context:2 ~seed:1 in
      assert_vec [| 0.; 0.; 1.; 1.; 0.; 0. |] (Neural.input_of_context m [| 2; 0 |]) );
    ( "input_of_context wrong length raises" >:: fun _ ->
      let m = Neural.create ~vocab_size:3 ~context:2 ~seed:1 in
      raises_invalid (fun () -> Neural.input_of_context m [| 1 |]) );
    ( "input_of_context bad id raises" >:: fun _ ->
      let m = Neural.create ~vocab_size:3 ~context:2 ~seed:1 in
      raises_invalid (fun () -> Neural.input_of_context m [| 1; 3 |]) );
    ( "context 0 predicts from the bias alone" >:: fun _ ->
      let m = Neural.of_params ~context:0 ~weights:[| [||]; [||] |] ~bias:[| 0.; log 3. |] in
      assert_vec [| 0.25; 0.75 |] (Neural.predict m [||]) );
  ]

(* ---------- Neural: exact predictions and loss ---------- *)

(* Vocab 2, context 1. Column 0 of W is used for context [|0|] and column 1
   for context [|1|].
   - After [|0|]: scores = [0; ln 3], so p = [1/4; 3/4].
   - After [|1|]: scores = [0; 0], so p = [1/2; 1/2]. *)
let hand_model () =
  Neural.of_params ~context:1
    ~weights:[| [| 0.; 0. |]; [| log 3.; 0. |] |]
    ~bias:[| 0.; 0. |]

let exact_tests =
  [
    ( "predict after id 0" >:: fun _ ->
      assert_vec [| 0.25; 0.75 |] (Neural.predict (hand_model ()) [| 0 |]) );
    ( "predict after id 1" >:: fun _ ->
      assert_vec [| 0.5; 0.5 |] (Neural.predict (hand_model ()) [| 1 |]) );
    (* bias [ln 3; 0] adds to scores [0; 0], so p = [3/4; 1/4] *)
    ( "bias shifts the prediction" >:: fun _ ->
      let m =
        Neural.of_params ~context:1 ~weights:(Matrix.zeros 2 2) ~bias:[| log 3.; 0. |]
      in
      assert_vec [| 0.75; 0.25 |] (Neural.predict m [| 1 |]) );
    (* -ln (3/4) = ln (4/3) *)
    ( "example_loss" >:: fun _ ->
      assert_equal ~printer:string_of_float ~cmp:(close ~eps:1e-9) (log (4. /. 3.))
        (Neural.example_loss (hand_model ()) (ex [| 0 |] 1)) );
    (* (ln (4/3) + ln 2) / 2 *)
    ( "average_loss" >:: fun _ ->
      assert_equal ~printer:string_of_float ~cmp:(close ~eps:1e-9)
        ((log (4. /. 3.) +. log 2.) /. 2.)
        (Neural.average_loss (hand_model ()) [ ex [| 0 |] 1; ex [| 1 |] 0 ]) );
    ( "example_loss bad target raises" >:: fun _ ->
      raises_invalid (fun () -> Neural.example_loss (hand_model ()) (ex [| 0 |] 2)) );
    ( "average_loss of no examples raises" >:: fun _ ->
      raises_invalid (fun () -> Neural.average_loss (hand_model ()) []) );
    (* every prediction starts near 1/V, so the loss starts near ln V *)
    ( "a new model's loss is about ln V" >:: fun _ ->
      let m = Neural.create ~vocab_size:5 ~context:2 ~seed:3 in
      let exs = [ ex [| 0; 1 |] 2; ex [| 3; 4 |] 0; ex [| 2; 2 |] 4 ] in
      assert_equal ~printer:string_of_float ~cmp:(close ~eps:0.05) (log 5.)
        (Neural.average_loss m exs) );
  ]

(* ---------- Neural: gradient check ---------- *)

(* Compare the change [train_step] makes with a numerical estimate of the
   gradient. With learning rate 1, the step changes each weight by minus its
   gradient. The estimate nudges one weight up and down by [h] and measures
   how the loss changes: (loss(w + h) - loss(w - h)) / 2h. *)
let gradient_check_test =
  "train_step follows the numerical gradient" >:: fun _ ->
  let m = Neural.create ~vocab_size:3 ~context:1 ~seed:11 in
  let example = ex [| 1 |] 2 in
  let w0 = Neural.weights m and b0 = Neural.bias m in
  let stepped = Neural.of_params ~context:1 ~weights:w0 ~bias:b0 in
  Neural.train_step stepped ~learning_rate:1. example;
  let w1 = Neural.weights stepped and b1 = Neural.bias stepped in
  let h = 1e-5 in
  let loss_with w b = Neural.example_loss (Neural.of_params ~context:1 ~weights:w ~bias:b) example in
  for j = 0 to 2 do
    for k = 0 to 2 do
      let up = Matrix.copy w0 and down = Matrix.copy w0 in
      up.(j).(k) <- up.(j).(k) +. h;
      down.(j).(k) <- down.(j).(k) -. h;
      let numeric = (loss_with up b0 -. loss_with down b0) /. (2. *. h) in
      let analytic = w0.(j).(k) -. w1.(j).(k) in
      assert_equal
        ~msg:(Printf.sprintf "weight %d %d" j k)
        ~printer:string_of_float ~cmp:(close ~eps:1e-6) numeric analytic
    done;
    let up = Array.copy b0 and down = Array.copy b0 in
    up.(j) <- up.(j) +. h;
    down.(j) <- down.(j) -. h;
    let numeric = (loss_with w0 up -. loss_with w0 down) /. (2. *. h) in
    assert_equal
      ~msg:(Printf.sprintf "bias %d" j)
      ~printer:string_of_float ~cmp:(close ~eps:1e-6) numeric (b0.(j) -. b1.(j))
  done

(* ---------- Neural: learning ---------- *)

(* fixtures/abc.txt is "abcabcabd", so the vocabulary is a b c d (ids 0..3).
   With context 2 the examples are:
     ab->c  bc->a  ca->b  ab->c  bc->a  ca->b  ab->d
   so the best possible model has p(c | ab) = 2/3, p(d | ab) = 1/3,
   p(a | bc) = 1 and p(b | ca) = 1. *)
let abc_examples () =
  let corpus = get_ok (Corpus.load "fixtures/abc.txt") in
  get_ok (Corpus.examples corpus ~context:2)

let learning_tests =
  [
    ( "training lowers the loss" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      let losses = Neural.train m ~learning_rate:0.5 ~epochs:50 (abc_examples ()) in
      assert_equal 50 (List.length losses);
      assert_bool "last < first" (List.nth losses 49 < List.hd losses) );
    ( "learns the counts in abc.txt" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      ignore (Neural.train m ~learning_rate:0.1 ~epochs:1000 (abc_examples ()));
      let after_ab = Neural.predict m [| 0; 1 |] in
      assert_equal ~msg:"p(c | ab)" ~printer:string_of_float ~cmp:(close ~eps:0.1)
        (2. /. 3.) after_ab.(2);
      assert_equal ~msg:"p(d | ab)" ~printer:string_of_float ~cmp:(close ~eps:0.1)
        (1. /. 3.) after_ab.(3);
      assert_bool "p(a | bc) > 0.9" ((Neural.predict m [| 1; 2 |]).(0) > 0.9) );
    ( "zero epochs leaves the model alone" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      let before = Neural.weights m in
      assert_equal [] (Neural.train m ~learning_rate:0.5 ~epochs:0 (abc_examples ()));
      assert_equal ~printer:print_mat before (Neural.weights m) );
    ( "train with no examples raises" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      raises_invalid (fun () -> Neural.train m ~learning_rate:0.5 ~epochs:1 []) );
  ]

(* ---------- Neural: generation ---------- *)

let all_in_range v ids = Array.for_all (fun id -> id >= 0 && id < v) ids

let generation_tests =
  [
    ( "generate gives the requested number of ids in range" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      let ids = Neural.generate m (Random.State.make [| 1 |]) ~prompt:[| 0; 1 |] ~length:30 in
      assert_equal 30 (Array.length ids);
      assert_bool "in range" (all_in_range 4 ids) );
    ( "generate is reproducible" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      let run () = Neural.generate m (Random.State.make [| 9 |]) ~prompt:[| 0; 1 |] ~length:20 in
      assert_equal (run ()) (run ()) );
    ( "generate length 0" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      assert_equal [||] (Neural.generate m (Random.State.make [| 1 |]) ~prompt:[| 0; 1 |] ~length:0)
    );
    ( "generate with a short prompt raises" >:: fun _ ->
      let m = Neural.create ~vocab_size:4 ~context:2 ~seed:1 in
      raises_invalid (fun () ->
          Neural.generate m (Random.State.make [| 1 |]) ~prompt:[| 0 |] ~length:5) );
    (* a model that is sure the next id is 1 must always sample 1 *)
    ( "sample_next follows a certain prediction" >:: fun _ ->
      let m =
        Neural.of_params ~context:0 ~weights:[| [||]; [||]; [||] |] ~bias:[| -100.; 100.; -100. |]
      in
      let st = Random.State.make [| 4 |] in
      for _ = 1 to 50 do
        assert_equal 1 (Neural.sample_next m st [||])
      done );
  ]

(* ---------- Integration with Corpus and Vocab ---------- *)

let integration_tests =
  [
    ( "load, train, generate, decode" >:: fun _ ->
      let corpus = get_ok (Corpus.load "fixtures/abc.txt") in
      let vocab = Corpus.vocab corpus in
      let exs = get_ok (Corpus.examples corpus ~context:2) in
      let m = Neural.create ~vocab_size:(Vocab.size vocab) ~context:2 ~seed:1 in
      ignore (Neural.train m ~learning_rate:0.5 ~epochs:200 exs);
      let prompt = Array.sub (Corpus.ids corpus) 0 2 in
      let text = Vocab.decode vocab (Neural.generate m (Random.State.make [| 1 |]) ~prompt ~length:20) in
      assert_equal 20 (String.length text);
      assert_bool "only corpus characters" (String.for_all (Vocab.mem vocab) text) );
  ]

let () =
  run_test_tt_main
    ("ocamlgpt neural"
    >::: [
           "matrix" >::: matrix_tests;
           "matrix properties" >::: matrix_properties;
           "neural shape" >::: shape_tests;
           "neural exact" >::: exact_tests;
           "neural gradient" >::: [ gradient_check_test ];
           "neural learning" >::: learning_tests;
           "neural generation" >::: generation_tests;
           "integration" >::: integration_tests;
         ])
