(lang dune 3.0)

(name ocamlgpt)

(generate_opam_files true)

(authors
 "Matt Zhu <hz754@cornell.edu>"
 "Matheo Muecke <mem572@cornell.edu>"
 "Nicholas Arana <nea43@cornell.edu>"
 "Alejandro Lilienthal <al2594@cornell.edu>")

(package
 (name ocamlgpt)
 (synopsis "OcamlGPT: small character-level language models built from scratch in OCaml")
 (depends ocaml dune ounit2 qcheck-core qcheck-ounit))
