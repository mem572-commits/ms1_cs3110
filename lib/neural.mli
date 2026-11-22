(** A small neural language model over characters.

    The model predicts the next character id from the [context] ids before
    it. It has one layer:

    - The input [x] is the one-hot vectors of the context ids, joined end to
      end, so it has length [context * vocab_size]. See {!input_of_context}.
    - The scores are [W x + b], where [W] is a [vocab_size] by
      [context * vocab_size] matrix and [b] is a vector of length
      [vocab_size].
    - The prediction is [softmax (W x + b)]: one probability for each
      possible next id.

    Training uses stochastic gradient descent on the cross-entropy loss
    [-. log p.(target)], one example at a time (see {!train_step}).

    The weights are updated in place: {!train_step} and {!train} change the
    model they are given. Everything else leaves the model unchanged.

    Passing a context of the wrong length, an id outside
    [0 .. vocab_size - 1], or other sizes that do not fit raises
    [Invalid_argument]. User-input problems, such as a missing file or a
    text that is too short, are handled before this module by {!Corpus}. *)

type t
(** A model with a fixed vocabulary size and context length. *)

(** {1 Construction} *)

val create : vocab_size:int -> context:int -> seed:int -> t
(** [create ~vocab_size ~context ~seed] is a new model whose weights are
    small random numbers (in [[-0.01, 0.01]]) chosen from [seed], and
    whose bias is zero. Its predictions start close to uniform. The same
    seed always gives the same model.
    Raises [Invalid_argument] if [vocab_size < 1] or [context < 0]. *)

val of_params : context:int -> weights:Matrix.mat -> bias:Matrix.vec -> t
(** [of_params ~context ~weights ~bias] is the model with exactly these
    weights and bias. Its vocabulary size is [Array.length bias]. The
    arrays are copied, so changing them later does not change the model.
    This is useful for tests and for loading saved parameters.
    Raises [Invalid_argument] if [bias] is empty, [context < 0], or
    [weights] is not [Array.length bias] by [context * Array.length bias]. *)

(** {1 Queries} *)

val vocab_size : t -> int
(** [vocab_size m] is the number of possible ids. *)

val context_length : t -> int
(** [context_length m] is the number of ids the model looks at. *)

val weights : t -> Matrix.mat
(** [weights m] is a copy of the weight matrix of [m]. *)

val bias : t -> Matrix.vec
(** [bias m] is a copy of the bias vector of [m]. *)

(** {1 Prediction} *)

val input_of_context : t -> int array -> Matrix.vec
(** [input_of_context m ctx] is the one-hot vectors of the ids in [ctx],
    joined end to end. Example: with [vocab_size = 3], the context
    [[|2; 0|]] gives [[|0.; 0.; 1.; 1.; 0.; 0.|]].
    Raises [Invalid_argument] if [Array.length ctx <> context_length m] or
    any id is out of range. *)

val predict : t -> int array -> Matrix.vec
(** [predict m ctx] is the model's probability for each possible next id
    after [ctx]. It has length [vocab_size m] and adds up to [1.].
    Raises [Invalid_argument] as {!input_of_context} does. *)

val example_loss : t -> Corpus.example -> float
(** [example_loss m ex] is [-. log p.(ex.target)], where
    [p = predict m ex.context]. It is [0.] for a perfect prediction and
    larger for worse ones.
    Raises [Invalid_argument] if the context does not fit or the target is
    out of range. *)

val average_loss : t -> Corpus.example list -> float
(** [average_loss m exs] is the mean of {!example_loss} over [exs].
    Raises [Invalid_argument] if [exs] is empty. *)

(** {1 Training} *)

val train_step : t -> learning_rate:float -> Corpus.example -> unit
(** [train_step m ~learning_rate ex] moves the weights and bias of [m] one
    step of size [learning_rate] downhill on [example_loss m ex].
    Raises [Invalid_argument] as {!example_loss} does. *)

val train :
  t -> learning_rate:float -> epochs:int -> Corpus.example list -> float list
(** [train m ~learning_rate ~epochs exs] runs [epochs] passes over [exs], in
    order, calling {!train_step} on each example. The result is the
    {!average_loss} on [exs] after each pass, so it has length [epochs].
    Raises [Invalid_argument] if [epochs < 0] or [exs] is empty. *)

(** {1 Generation} *)

val sample_next : t -> Random.State.t -> int array -> int
(** [sample_next m st ctx] picks a next id at random using [st], with each
    id chosen with probability [(predict m ctx).(id)].
    Raises [Invalid_argument] as {!predict} does. *)

val generate : t -> Random.State.t -> prompt:int array -> length:int -> int array
(** [generate m st ~prompt ~length] is [length] new ids that follow
    [prompt]. Each id is chosen by {!sample_next} from the last
    [context_length m] ids so far. The prompt itself is not included in the
    result. The same state gives the same output.
    Raises [Invalid_argument] if [length < 0], the prompt is shorter than
    [context_length m], or the prompt has an id out of range. *)
