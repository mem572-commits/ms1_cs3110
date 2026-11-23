
(* AF: [{ context; vocab_size; counts }] is the n-gram model that looks at
   the last [context] ids. For a context [ctx], [counts] maps [ctx] to an
   array [a] where [a.(i)] is the number of times id [i] followed [ctx] in
   the training text. Contexts that never appeared are not in [counts].

   RI: [context >= 0] and [vocab_size >= 1].
   Every key in [counts] has length [context], and its ids are in
   [0 .. vocab_size - 1].
   Every value in [counts] has length [vocab_size], all its entries are
   [>= 0], and at least one entry is [> 0]. *)

type t = {
    context : int;
    vocab_size : int;
    counts : (int array, int array) Hashtbl.t;
}

let fit corpus ~context = 
    match (Corpus.examples_seq corpus ~context) with 
    | Error e -> Error e
    | Ok examples -> 
        let vocab_size = Vocab.size (Corpus.vocab corpus) in 
        let counts = Hashtbl.create 1024 in 
        Seq.iter 
            (fun { Corpus.context = ctx; target } ->
            let row = 
                match (Hashtbl.find_opt counts ctx) with
                | Some row -> row
                | None -> 
                    let row = Array.make vocab_size 0 in
                    Hashtbl.add counts ctx row;
                    row
            in 
            row.(target) <- row.(target) + 1)
            examples;
        Ok { context; vocab_size; counts }

let vocab_size m = m.vocab_size
let context_length m = m.context

let count m ctx id =
  if Array.length ctx <> m.context then
    invalid_arg
      (Printf.sprintf "Ngram.count: context has length %d but the model uses %d"
         (Array.length ctx) m.context)
  else if id < 0 || id >= m.vocab_size then
    invalid_arg
      (Printf.sprintf "Ngram.count: id %d is not in 0..%d" id (m.vocab_size - 1))
  else
    match Hashtbl.find_opt m.counts ctx with
    | None -> 0
    | Some row -> row.(id)

let predict m ctx =
  if Array.length ctx <> m.context then
    invalid_arg
      (Printf.sprintf "Ngram.predict: context has length %d but the model uses %d"
         (Array.length ctx) m.context)
  else
    match Hashtbl.find_opt m.counts ctx with
    | None -> Array.make m.vocab_size (1. /. float m.vocab_size)
    | Some row ->
        let total = Array.fold_left ( + ) 0 row in
        Array.map (fun c -> float c /. float total) row

let sample_next m st ctx =
  let p = predict m ctx in
  let u = Random.State.float st 1.0 in
  let rec go i acc =
    if i = Array.length p - 1 then i
    else
      let acc' = acc +. p.(i) in
      if u < acc' then i else go (i + 1) acc'
  in
  go 0 0.

let generate m st ~prompt ~length = 
    if Array.length prompt < m.context then 
        invalid_arg
            (Printf.sprintf "Ngram.generate: prompt length must be at least the context length %d."
            m.context )
    else if length < 0 then
        invalid_arg
            (Printf.sprintf "Ngram.generate: length must be greater than or equal to 0 but is %d."
            length )
    else 
        let out = Array.make length 0 in 
        let ctx = ref (Array.sub prompt (Array.length prompt - m.context) m.context) in
        for i = 0 to length - 1 do
        let next = sample_next m st !ctx in
        out.(i) <- next;
        if m.context > 0 then
            ctx := Array.append (Array.sub !ctx 1 (m.context - 1)) [| next |]
        done;
        out
