(** Character vocabularies.

    A vocabulary is a fixed, one-to-one mapping between a set of characters
    and the integer ids [0 .. size v - 1]. Models never see characters; they
    see ids. Use {!encode} to turn text into ids and {!decode} to turn ids
    back into text.

    Ids are assigned in increasing character-code order, so building a
    vocabulary from the same characters always gives the same ids, no matter
    what order the characters appear in. This keeps tests and seeded runs
    reproducible.

    A "character" here is an OCaml [char], which is one byte. Plain ASCII
    text works as expected. A UTF-8 character such as ['é'] is stored as
    several bytes, and each byte gets its own id. Encoding and then decoding
    still gives back exactly the original text. *)

type t
(** A character vocabulary. Values of this type are immutable. *)

exception Unknown_char of char
(** Raised by {!encode} and {!encode_char} when a character is not in the
    vocabulary. *)

(** {1 Construction} *)

val of_string : string -> t
(** [of_string s] is the vocabulary of the distinct characters in [s].
    Example: [of_string "hello"] has the characters ['e'; 'h'; 'l'; 'o'] with
    ids [0; 1; 2; 3]. [of_string ""] is the empty vocabulary. *)

val of_chars : char list -> t
(** [of_chars cs] is the vocabulary of the distinct characters in [cs].
    Duplicates are ignored. *)

val serialize : t -> string
(** [serialize v] is a string holding each character of [v] once, in id
    order. [of_string (serialize v)] is equal to [v], so this string can be
    used to save a vocabulary alongside saved model parameters. *)

(** {1 Queries} *)

val size : t -> int
(** [size v] is the number of characters in [v]. Valid ids are
    [0 .. size v - 1]. *)

val chars : t -> char list
(** [chars v] lists the characters of [v] in id order. *)

val mem : t -> char -> bool
(** [mem v c] is [true] if [c] is in [v]. *)

val id_of_char : t -> char -> int option
(** [id_of_char v c] is [Some i] if [c] has id [i] in [v], or [None] if [c]
    is not in [v]. *)

val char_of_id : t -> int -> char option
(** [char_of_id v i] is [Some c] if id [i] stands for [c] in [v], or [None]
    if [i] is out of range. *)

val equal : t -> t -> bool
(** [equal a b] is [true] if [a] and [b] contain the same characters, and so
    give every character the same id. *)

(** {1 Encoding and decoding} *)

val encode_char : t -> char -> int
(** [encode_char v c] is the id of [c] in [v].
    Raises [Unknown_char c] if [c] is not in [v]. *)

val decode_id : t -> int -> char
(** [decode_id v i] is the character with id [i] in [v].
    Raises [Invalid_argument] if [i] is out of range. *)

val encode : t -> string -> int array
(** [encode v s] is the ids of the characters of [s], in order. The result
    has length [String.length s].
    Raises [Unknown_char c] for the first character [c] of [s] that is not in
    [v]. Use {!try_encode} to get a result instead of an exception. *)

val try_encode : t -> string -> (int array, char list) result
(** [try_encode v s] is [Ok (encode v s)] if every character of [s] is in
    [v]. Otherwise it is [Error cs], where [cs] lists each unknown character
    once, in the order it first appears in [s]. *)

val decode : t -> int array -> string
(** [decode v ids] is the text made of the characters for [ids], in order.
    [decode v (encode v s) = s] for any [s] whose characters are all in [v].
    Raises [Invalid_argument] if any id is out of range. *)

val unknown_chars : t -> string -> char list
(** [unknown_chars v s] lists the characters of [s] that are not in [v],
    each once, in the order they first appear in [s]. It is [[]] if [s] can
    be encoded. *)

(** {1 Printing} *)

val display_char : char -> string
(** [display_char c] is a readable form of [c] for messages: ["\\n"] for a
    newline, ["\\t"] for a tab, ["' '"] for a space, ["\\xNN"] for other
    non-printable bytes, and the character itself otherwise. *)

val to_string : t -> string
(** [to_string v] is a readable listing of the characters of [v] in id
    order, for example [{e h l o}]. *)
