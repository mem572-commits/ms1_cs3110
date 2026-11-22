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

let () =
  run_test_tt_main
    ("ocamlgpt neural"
    >::: [ "matrix" >::: matrix_tests; "matrix properties" >::: matrix_properties ])
