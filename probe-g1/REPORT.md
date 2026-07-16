# G1 probe report — discharge automation for KIR certificates

*2026-07-16. The DESIGN.md §5.2 gate measurement. Verdict up front:*

## Verdict: gate PASSED

**37 verified, 0 errors**, under the tactus Lean backend (`./check.sh`), closer-free
(zero `tactus_tactic` strings), zero `assume`/`admit`/`external_body`. Both
certificate lemmas — the eight-statement step bisimulation and the loop
induction — verified on the **first attempt** once the schema below was in
place. Warm re-check: 1.4 s.

The gate asked: is bisimulation discharge mostly-automatic on a loop + carry
chain + locals, with any needed tactics being kernel-independent schema? The
answer is yes, stronger than required.

## What was proved

`certificate.rs::lemma_loop_bisim`: running the KIR loop (counter-writing
runner + eight-statement body: two buffer reads, two wrapping adds, two
compare-bits, one buffer write, carry update) from any well-formed state
equals the model loop — output buffers equal, final carry equal, widths
preserved — by induction on the trip count, with full-state-equality
step lemmas underneath. The model mirrors the u32 compare-trick source shape
(what the SST reflection will stand for); the KIR literal is what the
`--emit-gpu` pass would emit.

## The discharge schema (what the generator emits)

Every proof in the probe falls into one of five mechanical shapes:

1. **`u_*` one-step unfold lemmas — empty bodies, 28 of them, 100% automatic.**
   Recursive spec fns (interpreters, model loops) are opaque to `simp`; their
   per-constructor equations must be provided as callable lemmas. The default
   closer proves all of them with no body at all.
2. **Per-statement lemmas (8): fixed pattern** — three or four `u_*` calls, one
   cast bridge (`push_cast` + `omega`), one connective assert
   (`simp_all (config := { zetaDelta := true }) [<the statement's spec fn>]`).
3. **Seq bookkeeping: explicit vstd axiom calls** —
   `vstd::seq::axiom_seq_update_{len,same,different}`, 25 calls for the
   8-statement body, every one determined by the write-index sequence
   (computable by the generator from the KIR literal alone).
4. **State-chain asserts: uniform** — every one is
   `simp_all (config := { zetaDelta := true }) [addloop.mk_state]` or the
   same with an empty lemma list.
5. **One arithmetic seam** — the source's plain `c1 + c2` (proven in-range)
   vs KIR's wrapping `AddW`: closed by
   `simp_all … [wadd, ltu] <;> split_ifs <;> (try push_cast at *) <;> omega`.
   This is the standard block for any in-range-add vs wrapping-add seam, the
   one op-mapping mismatch the lowering introduces by design.

Total tactic vocabulary across the whole probe: `simp_all` with `zetaDelta`
and at most two names, `push_cast`, `omega`, `split_ifs`, `intros`. Nothing
kernel-specific, no proof search, no creativity.

## Discovered constraints (bind the G1 pass design)

- **`#[verifier::tactus_auto]` on proof fns is load-bearing**: without the
  attribute, Lean-tactic `assert … by { }` blocks fail to *parse* (the block
  is read as Verus proof code). The generator must emit the attribute.
- **Recursive spec fns get no usable `simp` equations** — the `u_*` unfold
  layer is mandatory, not stylistic. It is also free (empty bodies close).
- **vstd Seq axioms are callable and sufficient**; no Lean-side Seq lemma
  names needed anywhere.
- Proof-fn recursion with numeric `decreases` works as the induction vehicle;
  the termination VC closed automatically.

## Probe simplifications (honest scoping)

Not measured here, in expected-risk order:

1. **The source side is a hand-written model**, not SST reflection. The full
   certificate chains `source exec fn == model == KIR`. The left link is
   standard tactus exec verification (validated at scale by the
   tactus-group-theory migration); the *generation* of the model from SST is
   fork plumbing (sibling of DESIGN-bootstrap R2), not a discharge risk. This
   is the remaining G1 unknown, and it is an engineering unknown, not a
   proof-automation one.
2. **No buffer offsets** (`a_off`/`out_off`): adds index arithmetic to the
   cast bridges — more `omega`, same schema.
3. **Totalized eval, result-equality only**: production KIR carries `ksafe`
   side conditions (DESIGN §4.2) as additional certificate conjuncts — same
   facts the equality proof already establishes along the way.
4. **Single loop, no functions**: multi-loop kernels compose the same
   step/loop lemma pairs; KIR function calls will need one more `u_` unfold
   shape (call frame), no new mechanism apparent.

## Cost model

Certificate size scales linearly with statement count (per-statement lemma +
~3 Seq axiom calls per write). The 8-statement body cost ~450 lines of fully
mechanical proof — the generator emits it; nobody writes it. Verification
time is unremarkable (cold run ~1 min for the whole crate, warm 1.4 s,
function-level cache applies).

## Recommendation

Proceed with G1 proper: the `--emit-gpu` pass in the fork, emitting exactly
this schema. Start with straight-line kernels (protolith `sum_sq3`,
`wrap_delta`, `torus_dist2`), then the loop template. The probe's
`certificate.rs` is the golden reference for what generated output should
look like.

---

# Addendum — G1a: the generator exists (2026-07-16, same day)

**`kirgen/` (plain Rust, zero deps, one reviewable file) generated
`probe-g1/src/gen_certs.rs` (536 lines), and the crate went to
52 verified, 0 errors on the first generation run — no hand edits to
generated output, no template tuning.**

Kernels certified (real protolith kernel shapes, certificates target the
verbatim spec fns): `abs_delta` (Select + SubW), `wrap_delta` (three
statements, nested Select, the n−d seam), `sum_sq3` (MulW with generated
nlinarith range-bound asserts). New discharge shapes beyond the probe —
Select, wrapping sub/mul, generated seam blocks — all closed by the same
five-shape schema with `nlinarith` added to the arithmetic cascade.

**Design discovery worth recording:** for kernels carrying exact functional
postconditions (every protolith kernel does — house style), the certificate
targets the *spec fn*, and the chain `exec == spec` (already verified in the
source crate) `== KIR` (generated) is complete **without SST reflection**.
Reflection is thereby demoted from G1 blocker to an optimization for kernels
lacking functional specs. The kernel-safe subset (DESIGN §4.1) should simply
require exact functional postconditions.

The generator's input seam (typed kernel AST, hand-encoded in `main.rs`) is
what the fork's SST adapter replaces. Transcription errors at this seam are
caught by verification: every certificate's ensures references the real
`spec_kernels::*` fn by name, so a mis-encoded kernel fails its own
certificate. (Residual: contradictory `requires` would make a certificate
vacuous — the requires mirror protolith's visibly-satisfiable preconditions;
a generated witness-instantiation check is cheap future hardening.)

Remaining for full G1: loop-kernel emission (schema hand-proven in
`certificate.rs`, needs templating into kirgen), the SST adapter in the fork,
cross-crate generation (certificates importing protolith directly).
