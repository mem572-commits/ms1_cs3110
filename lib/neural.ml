(* AF: [{ context; vocab_size; weights; bias }] is the model that predicts
   [softmax (weights * x + bias)], where [x] is the one-hot encoding of the last
   [context] ids, joined end to end.

   RI: [vocab_size >= 1] and [context >= 0]. [weights] has [vocab_size] rows,
   each of length [context * vocab_size]. [bias] has length [vocab_size].
   [weights] and [bias] are never shared outside this module. *)
type t = {
  context : int;
  vocab_size : int;
  weights : Matrix.mat;
  bias : Matrix.vec;
}

(* ---------- Construction ---------- *)

let create ~vocab_size ~context ~seed =
  if vocab_size < 1 then
    invalid_arg "Neural.create: vocab_size must be at least 1";
  if context < 0 then invalid_arg "Neural.create: context must be 0 or more";
  let st = Random.State.make [| seed |] in
  {
    context;
    vocab_size;
    weights = Matrix.random st vocab_size (context * vocab_size) ~scale:0.01;
    bias = Array.make vocab_size 0.;
  }

let of_params ~context ~weights ~bias =
  let v = Array.length bias in
  if v = 0 then invalid_arg "Neural.of_params: bias is empty";
  if context < 0 then invalid_arg "Neural.of_params: context must be 0 or more";
  if
    Matrix.rows weights <> v
    || Array.exists (fun row -> Array.length row <> context * v) weights
  then
    invalid_arg
      (Printf.sprintf "Neural.of_params: weights must be %d x %d" v (context * v));
  {
    context;
    vocab_size = v;
    weights = Matrix.copy weights;
    bias = Array.copy bias;
  }

(* ---------- Queries ---------- *)

let vocab_size m = m.vocab_size
let context_length m = m.context
let weights m = Matrix.copy m.weights
let bias m = Array.copy m.bias

(* ---------- Prediction ---------- *)

let check_id name m id =
  if id < 0 || id >= m.vocab_size then
    invalid_arg
      (Printf.sprintf "Neural.%s: id %d is not in 0..%d" name id
         (m.vocab_size - 1))

let input_of_context m ctx =
  if Array.length ctx <> m.context then
    invalid_arg
      (Printf.sprintf
         "Neural.input_of_context: context has length %d, expected %d"
         (Array.length ctx) m.context);
  let x = Array.make (m.context * m.vocab_size) 0. in
  Array.iteri
    (fun i id ->
      check_id "input_of_context" m id;
      x.((i * m.vocab_size) + id) <- 1.)
    ctx;
  x

let probs_of_input m x =
  Matrix.softmax (Matrix.add_vec (Matrix.mat_vec m.weights x) m.bias)

let predict m ctx = probs_of_input m (input_of_context m ctx)

let example_loss m { Corpus.context; target } =
  let p = predict m context in
  check_id "example_loss" m target;
  -.log p.(target)

let average_loss m exs =
  match exs with
  | [] -> invalid_arg "Neural.average_loss: no examples"
  | _ ->
      let total =
        List.fold_left (fun acc ex -> acc +. example_loss m ex) 0. exs
      in
      total /. Float.of_int (List.length exs)

(* ---------- Training ---------- *)

(* For the loss [-. log p.(target)] with [p = softmax (W x + b)], the gradient
   of the score for id [j] is [p.(j) - 1] if [j = target] and [p.(j)] otherwise.
   The gradient of [W.(j).(k)] is that times [x.(k)], and the gradient of
   [b.(j)] is that alone. *)
let train_step m ~learning_rate { Corpus.context; target } =
  let x = input_of_context m context in
  check_id "train_step" m target;
  let p = probs_of_input m x in
  for j = 0 to m.vocab_size - 1 do
    let grad = if j = target then p.(j) -. 1. else p.(j) in
    let row = m.weights.(j) in
    for k = 0 to Array.length x - 1 do
      row.(k) <- row.(k) -. (learning_rate *. grad *. x.(k))
    done;
    m.bias.(j) <- m.bias.(j) -. (learning_rate *. grad)
  done

let train m ~learning_rate ~epochs exs =
  if epochs < 0 then invalid_arg "Neural.train: epochs must be 0 or more";
  if exs = [] then invalid_arg "Neural.train: no examples";
  List.init epochs (fun _ ->
      List.iter (train_step m ~learning_rate) exs;
      average_loss m exs)

(* ---------- Generation ---------- *)

(* Walk through the probabilities until their running total passes a random
   number in [0, 1). If rounding leaves the total just below the number, the
   last id is chosen. *)
let sample_next m st ctx =
  let p = predict m ctx in
  let r = Random.State.float st 1. in
  let rec pick i total =
    if i = Array.length p - 1 then i
    else
      let total = total +. p.(i) in
      if r < total then i else pick (i + 1) total
  in
  pick 0 0.

let generate m st ~prompt ~length =
  if length < 0 then invalid_arg "Neural.generate: length must be 0 or more";
  let n = Array.length prompt in
  if n < m.context then
    invalid_arg
      (Printf.sprintf
         "Neural.generate: prompt has %d ids but the context needs %d" n
         m.context);
  Array.iter (check_id "generate" m) prompt;
  let ids = Array.append prompt (Array.make length 0) in
  for i = n to n + length - 1 do
    ids.(i) <- sample_next m st (Array.sub ids (i - m.context) m.context)
  done;
  Array.sub ids n length
