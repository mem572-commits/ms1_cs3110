(** Vectors and matrices of floats.

    These are the numerical building blocks for the neural model. The
    representation is deliberately plain: a vector is a [float array] and a
    matrix is an array of rows, each a [float array] of the same length.

    A matrix with [r] rows and [c] columns maps a vector of length [c] to a
    vector of length [r] (see {!mat_vec}).

    Functions that combine two values check their sizes and raise
    [Invalid_argument] with the function's name if they do not fit. No function
    changes its arguments; each returns a fresh array. *)

type vec = float array
(** A vector. *)

type mat = float array array
(** A matrix, stored row by row. Every row has the same length. *)

(** {1 Construction} *)

val zeros : int -> int -> mat
(** [zeros r c] is the [r] by [c] matrix of zeros. Raises [Invalid_argument] if
    [r < 0] or [c < 0]. *)

val random : Random.State.t -> int -> int -> scale:float -> mat
(** [random st r c ~scale] is an [r] by [c] matrix whose entries are drawn
    uniformly from [[-scale, scale]] using [st]. The same state gives the same
    matrix, so seeded runs are reproducible. Raises [Invalid_argument] if
    [r < 0], [c < 0], or [scale < 0.]. *)

val one_hot : int -> int -> vec
(** [one_hot n i] is the vector of length [n] that is [1.] at index [i] and [0.]
    everywhere else. Raises [Invalid_argument] if [i] is not in [0 .. n - 1]. *)

val copy : mat -> mat
(** [copy m] is a matrix equal to [m] that shares no rows with it. *)

(** {1 Queries} *)

val rows : mat -> int
(** [rows m] is the number of rows of [m]. *)

val cols : mat -> int
(** [cols m] is the number of columns of [m], or [0] if [m] has no rows. *)

(** {1 Operations} *)

val dot : vec -> vec -> float
(** [dot u v] is the sum of [u.(i) *. v.(i)] over all [i]. Raises
    [Invalid_argument] if the lengths differ. *)

val add_vec : vec -> vec -> vec
(** [add_vec u v] is the vector whose [i]-th entry is [u.(i) +. v.(i)]. Raises
    [Invalid_argument] if the lengths differ. *)

val mat_vec : mat -> vec -> vec
(** [mat_vec m v] is the matrix-vector product: the vector of length [rows m]
    whose [i]-th entry is [dot m.(i) v]. Raises [Invalid_argument] if
    [cols m <> Array.length v]. *)

val mat_mul : mat -> mat -> mat
(** [mat_mul a b] is the matrix product, of size [rows a] by [cols b]. Its entry
    at row [i], column [j] is the sum of [a.(i).(k) *. b.(k).(j)] over all [k].
    Raises [Invalid_argument] if [cols a <> rows b]. *)

val softmax : vec -> vec
(** [softmax v] is the vector of probabilities [exp v.(i) /. s], where [s] is
    the sum of [exp v.(j)] over all [j]. Every entry is positive and the entries
    add up to [1.]. The largest entry of [v] is subtracted before taking [exp],
    which gives the same answer but cannot overflow, so large inputs such as
    [[|1000.; 1000.|]] work. Raises [Invalid_argument] if [v] is empty. *)

val argmax : vec -> int
(** [argmax v] is the index of the largest entry of [v]. If several entries are
    equally large, it is the first of them. Raises [Invalid_argument] if [v] is
    empty. *)
