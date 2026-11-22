type vec = float array
type mat = float array array

(* ---------- Construction ---------- *)

let check_dims name r c =
  if r < 0 || c < 0 then
    invalid_arg (Printf.sprintf "Matrix.%s: negative size %d x %d" name r c)

let zeros r c =
  check_dims "zeros" r c;
  Array.make_matrix r c 0.

let random st r c ~scale =
  check_dims "random" r c;
  if scale < 0. then invalid_arg "Matrix.random: scale must be 0 or more";
  Array.init r (fun _ ->
      Array.init c (fun _ -> Random.State.float st (2. *. scale) -. scale))

let one_hot n i =
  if i < 0 || i >= n then
    invalid_arg (Printf.sprintf "Matrix.one_hot: index %d is not in 0..%d" i (n - 1));
  let v = Array.make n 0. in
  v.(i) <- 1.;
  v

let copy m = Array.map Array.copy m

(* ---------- Queries ---------- *)

let rows m = Array.length m
let cols m = if Array.length m = 0 then 0 else Array.length m.(0)

(* ---------- Operations ---------- *)

let check_same_length name u v =
  if Array.length u <> Array.length v then
    invalid_arg
      (Printf.sprintf "Matrix.%s: lengths %d and %d differ" name
         (Array.length u) (Array.length v))

let dot u v =
  check_same_length "dot" u v;
  let sum = ref 0. in
  for i = 0 to Array.length u - 1 do
    sum := !sum +. (u.(i) *. v.(i))
  done;
  !sum

let add_vec u v =
  check_same_length "add_vec" u v;
  Array.init (Array.length u) (fun i -> u.(i) +. v.(i))

let mat_vec m v =
  if cols m <> Array.length v then
    invalid_arg
      (Printf.sprintf "Matrix.mat_vec: matrix has %d columns but vector has length %d"
         (cols m) (Array.length v));
  Array.map (fun row -> dot row v) m

let mat_mul a b =
  if cols a <> rows b then
    invalid_arg
      (Printf.sprintf "Matrix.mat_mul: %d columns times %d rows" (cols a) (rows b));
  let n = cols a in
  Array.init (rows a) (fun i ->
      Array.init (cols b) (fun j ->
          let sum = ref 0. in
          for k = 0 to n - 1 do
            sum := !sum +. (a.(i).(k) *. b.(k).(j))
          done;
          !sum))

let argmax v =
  if Array.length v = 0 then invalid_arg "Matrix.argmax: empty vector";
  let best = ref 0 in
  for i = 1 to Array.length v - 1 do
    if v.(i) > v.(!best) then best := i
  done;
  !best

let softmax v =
  if Array.length v = 0 then invalid_arg "Matrix.softmax: empty vector";
  let biggest = v.(argmax v) in
  let exps = Array.map (fun x -> exp (x -. biggest)) v in
  let total = Array.fold_left ( +. ) 0. exps in
  Array.map (fun e -> e /. total) exps
