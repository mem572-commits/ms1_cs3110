
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