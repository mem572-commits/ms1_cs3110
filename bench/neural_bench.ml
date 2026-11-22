(* MS2 investigation for the neural model.

   Part 1 times Matrix.mat_mul on square matrices of a few sizes, to see
   whether plain OCaml arrays are fast enough for training.

   Part 2 trains Neural on a text file (data/sample.txt by default),
   printing the training and validation loss as it goes, then generates
   text from the trained model.

   Run from the repository root:
     dune exec bench/neural_bench.exe
     dune exec bench/neural_bench.exe -- path/to/file.txt *)

open Ocamlgpt

let context = 3
let epochs = 100
let learning_rate = 0.5
let seed = 3110

(* ---------- Part 1: matrix multiplication ---------- *)

let time f =
  let start = Sys.time () in
  let result = f () in
  (result, Sys.time () -. start)

let bench_mat_mul () =
  print_endline "== Matrix.mat_mul (n x n times n x n) ==";
  let st = Random.State.make [| seed |] in
  List.iter
    (fun n ->
      let a = Matrix.random st n n ~scale:1. in
      let b = Matrix.random st n n ~scale:1. in
      let _, secs = time (fun () -> Matrix.mat_mul a b) in
      Printf.printf "n = %4d: %.4f s\n" n secs)
    [ 50; 100; 200; 400 ];
  print_newline ()

(* ---------- Part 2: training on a text file ---------- *)

let fail e =
  prerr_endline ("Error: " ^ Corpus.error_to_string e);
  exit 1

let get = function
  | Ok x -> x
  | Error e -> fail e

let train_on path corpus =
  Printf.printf "== Training on %s ==\n" path;
  print_endline (Corpus.summary corpus);
  let train_part, valid_part = get (Corpus.split corpus ~train_ratio:0.9) in
  let train_exs = get (Corpus.examples train_part ~context) in
  let valid_exs = get (Corpus.examples valid_part ~context) in
  let vocab = Corpus.vocab corpus in
  let v = Vocab.size vocab in
  Printf.printf
    "Context %d, %d training and %d validation examples, learning rate %g\n"
    context (List.length train_exs) (List.length valid_exs) learning_rate;
  Printf.printf "Loss of a uniform guess: ln %d = %.3f\n\n" v (log (Float.of_int v));
  let model = Neural.create ~vocab_size:v ~context ~seed in
  let report epoch =
    Printf.printf "epoch %3d: train loss %.3f, validation loss %.3f\n" epoch
      (Neural.average_loss model train_exs)
      (Neural.average_loss model valid_exs)
  in
  report 0;
  let _, secs =
    time (fun () ->
        for epoch = 1 to epochs do
          ignore (Neural.train model ~learning_rate ~epochs:1 train_exs);
          if epoch mod 10 = 0 then report epoch
        done)
  in
  Printf.printf "\n%d epochs took %.3f s (%.5f s per epoch)\n\n" epochs secs
    (secs /. Float.of_int epochs);
  let prompt = Array.sub (Corpus.ids corpus) 0 context in
  let generated =
    Neural.generate model (Random.State.make [| seed |]) ~prompt ~length:200
  in
  Printf.printf "== 200 generated characters after %S ==\n%s%s\n"
    (Vocab.decode vocab prompt) (Vocab.decode vocab prompt)
    (Vocab.decode vocab generated)

let () =
  let path =
    match Sys.argv with
    | [| _ |] -> "data/sample.txt"
    | [| _; p |] -> p
    | _ ->
        prerr_endline "usage: neural_bench [training-file]";
        exit 2
  in
  (* Load first, so a bad file is reported before any benchmarking. *)
  let corpus = get (Corpus.load path) in
  bench_mat_mul ();
  train_on path corpus
