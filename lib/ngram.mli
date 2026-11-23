(** A character-level n-gram language model

    The model predicts the next id from the [context] ids before it, using how
    offer each id followed that context in the training text. Ids come from the
    corpus's {!Vocab.t}, so they match {!Neural}'s ids. *)

type t
(** A fitted model with a fixed vocabulary size and context length. Immutable
    once built. *)

(** {1 Construction} *)

val fit : Corpus.t -> context:int -> (t, Corpus.error) result
(** [fit c ~context] counts every (context, next id) pair in [c], using windows
    of [context] ids. Returns [Error (Invalid_context_length context)] if
    [context < 0], or [Error (Too_short _)] if [Corpus.length c <= context]. *)

(** {1 Queries} *)

val vocab_size : t -> int
(** [vocab_size m] is the number of possible ids. *)

val context_length : t -> int
(** [context_length m] is the number of ids the model looks at. *)

val count : t -> int array -> int -> int
(** [count m ctx id] is the number of times [id] followed [ctx] in the training
    text. Zero if [ctx] never appeared. Raises [Invalid_argument] if
    [Array.length ctx <> context_length m] or [id] is not in
    [0 .. vocab_size m - 1]. *)

(** {1 Prediction} *)

val predict : t -> int array -> Matrix.vec
(** [predict m ctx] is the probability of each possible next id after [ctx]: an
    array of length [vocab_size m] whose entries are in [0, 1] and sum to 1. If
    [ctx] never appeared in training, every id has probability
    [1 /. float (vocab_size m)]. Raises [Invalid_argument] if
    [Array.length ctx <> context_length m]. *)

(** {1 Generation} *)

val sample_next : t -> Random.State.t -> int array -> int
(** [sample_next m st ctx] picks a next id at random using [st], with each id
    chosen with its probability in [predict m ctx]. *)

val generate :
  t -> Random.State.t -> prompt:int array -> length:int -> int array
(** [generate m st ~prompt ~length] is [length] new ids that follow [prompt],
    each sampled from the last [context_length m] ids so far. The result does
    not include [prompt]. The same seed always gives the same result. Raises
    [Invalid_argument] if [Array.length prompt < context_length m] or
    [length < 0]. *)

(** {1 MS3 (not yet implemented)}

    - [average_loss : t -> Corpus.example list -> float], the mean of
      [-. log p.(target)], to compare with {!Neural.average_loss}.
    - interpolation for unseen contexts.
    - A temperature setting for sampling. *)
