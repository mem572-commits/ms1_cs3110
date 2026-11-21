(* AF: [{ chars; ids }] is the vocabulary in which character [chars.(i)] has
   id [i], for each [i] in [0 .. Array.length chars - 1].

   RI: [chars] holds distinct characters in strictly increasing code order.
   [ids] has length 256. For every character [c], [ids.(Char.code c) = i] if
   [chars.(i) = c], and [ids.(Char.code c) = -1] if [c] is not in [chars]. *)
type t = {
  chars : char array;
  ids : int array;
}

exception Unknown_char of char

let of_string s =
  let present = Array.make 256 false in
  String.iter (fun c -> present.(Char.code c) <- true) s;
  let ids = Array.make 256 (-1) in
  let found = ref [] in
  let next = ref 0 in
  Array.iteri
    (fun code here ->
      if here then begin
        ids.(code) <- !next;
        incr next;
        found := Char.chr code :: !found
      end)
    present;
  { chars = Array.of_list (List.rev !found); ids }

let of_chars cs = of_string (String.of_seq (List.to_seq cs))
let size v = Array.length v.chars
let serialize v = String.init (size v) (fun i -> v.chars.(i))
let chars v = Array.to_list v.chars
let mem v c = v.ids.(Char.code c) >= 0

let id_of_char v c =
  let i = v.ids.(Char.code c) in
  if i < 0 then None else Some i

let char_of_id v i = if i >= 0 && i < size v then Some v.chars.(i) else None
let equal a b = serialize a = serialize b

let encode_char v c =
  match id_of_char v c with
  | Some i -> i
  | None -> raise (Unknown_char c)

let decode_id v i =
  match char_of_id v i with
  | Some c -> c
  | None ->
      invalid_arg
        (Printf.sprintf "Vocab.decode_id: id %d is not in 0..%d" i (size v - 1))

let encode v s = Array.init (String.length s) (fun i -> encode_char v s.[i])
let decode v ids = String.init (Array.length ids) (fun i -> decode_id v ids.(i))

let unknown_chars v s =
  let seen = Array.make 256 false in
  let unknown = ref [] in
  String.iter
    (fun c ->
      let code = Char.code c in
      if (not (mem v c)) && not seen.(code) then begin
        seen.(code) <- true;
        unknown := c :: !unknown
      end)
    s;
  List.rev !unknown

let try_encode v s =
  match unknown_chars v s with
  | [] -> Ok (encode v s)
  | cs -> Error cs

let display_char = function
  | '\n' -> "\\n"
  | '\t' -> "\\t"
  | '\r' -> "\\r"
  | ' ' -> "' '"
  | c when Char.code c >= 33 && Char.code c <= 126 -> String.make 1 c
  | c -> Printf.sprintf "\\x%02x" (Char.code c)

let to_string v =
  "{" ^ String.concat " " (List.map display_char (chars v)) ^ "}"
