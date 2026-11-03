# V3 CPU Shadow Resume Design

## Intent

Build a free-CPU fallback that can make incremental progress on the DCT-v3 training question without pretending that CPU numerics are the already-frozen CUDA v3 experiment.

This is an **execution shadow lane**, not a replacement for frozen run `41580819667`.

## Doctrine checkpoint

CURRENT BEST: `two-stage-lora24-v1`, Public LB `0.520`, plain Gate-B OOF BA `0.5467611235396471`.

REAL EVIDENCE OF STRENGTH: v1 transferred to Public LB substantially better than earlier Phase7; v3 remains unrun.

MAIN FAILURE MODE: no usable GPU resource is currently attached to the frozen v3 transfer run.

CURRENT BOTTLENECK HYPOTHESIS: execution-resource scarcity, not yet evidence that DCT is scientifically bad.

IS THIS A LOCAL BUG OR MODEL-CLASS LIMITATION?: neither is established; current blocker is infrastructure.

WHAT METHOD CLASSES ARE CURRENTLY EXCLUDED?: CPU training was previously treated as operationally non-viable.

WHY ARE THEY EXCLUDED?: the exact-architecture CPU probe showed one worst-case triplet backward taking more than 90 minutes, and frozen CUDA uses fp16 autocast + GradScaler that CPU cannot reproduce exactly.

HAS THE ORIGINAL REASON FOR EXCLUSION CHANGED?: partially. The user explicitly prefers making slow free progress over waiting indefinitely for GPU, so CPU is reopened as shadow evidence, not authoritative v3 evidence.

SMALLEST EXPERIMENT THAT DISCRIMINATES BETWEEN THE COMPETING EXPLANATIONS: prove that a CPU training trajectory can stop mid-gradient-accumulation, persist only trainable state + optimizer/scheduler + RNG + partial gradients + cursor, resume in another process, and produce the exact same CPU result as uninterrupted execution.

WHAT RESULT WOULD CAUSE AN ARCHITECTURE-LEVEL PIVOT?: none. This lane is infrastructure. Scientific architecture decisions remain governed by transfer evidence.

## Restriction record

RESTRICTION: CPU shadow results cannot authorize v3 Gate B or a Kaggle submission.

WHY IT EXISTS: frozen v3 is CUDA fp16 autocast + GradScaler; CPU changes numerical backend and dropout/RNG trajectory relative to CUDA.

CURRENT EVIDENCE SUPPORTING IT: the actual frozen worker uses CUDA autocast, GradScaler, gradient clipping, AdamW, scheduler, LoRA dropout, head dropout, and groups six same-prompt triplets per optimizer step.

PIVOT TRIGGER: only a separately frozen contract may promote a CPU-specific candidate. CPU shadow evidence alone never retroactively becomes the frozen v3 gate.

STATUS: ACTIVE.

RESTRICTION: do not parallelize the six gradient-accumulation microbatches yet.

WHY IT EXISTS: independent workers would consume dropout RNG differently from the current sequential CPU trajectory unless per-microbatch RNG equivalence is proven.

CURRENT EVIDENCE SUPPORTING IT: LoRA dropout and head dropout are active during training.

PIVOT TRIGGER: a test must prove uninterrupted sequential CPU training and parallel/reduced-gradient training are bitwise-equivalent under an explicitly frozen RNG scheme.

STATUS: ACTIVE.

## Resume state

A resumable CPU checkpoint must persist:

- trainable LoRA parameters only, not frozen Qwen base weights;
- trainable LayerNorm/plain/centered head parameters;
- AdamW optimizer state;
- LR scheduler state;
- Python RNG state;
- NumPy RNG state;
- Torch CPU RNG state;
- partial accumulated gradients when a chunk ends inside a six-triplet optimizer group;
- optimizer-group cursor and microbatch cursor;
- loss/accounting counters;
- an immutable lineage fingerprint.

The frozen base model is reloaded from the exact model revision on every process and is not copied into checkpoints.

## Chunk semantics

A chunk may stop only after a completed microbatch backward. It may stop before `optimizer.step()` and preserve the partial gradients exactly.

A resumed process restores the checkpoint before computing the next microbatch. The next optimizer step must see the same accumulated gradients as an uninterrupted CPU process.

Default future Actions integration should run a deliberately small number of microbatches per job and upload one checkpoint at clean job exit. It must not rely on a 6-hour forced termination to preserve state.

## Scientific scope

Allowed:

- external Phase14 safe pool only;
- the already-frozen v3 transfer split;
- v1/v3 CPU shadow comparisons;
- timing and execution evidence;
- checkpoint/resume correctness evidence.

Forbidden:

- official competition rows;
- Public-LB tuning;
- changing DCT modes/layers/LoRA/data/source split based on shadow scores;
- treating CPU shadow PASS as frozen CUDA v3 PASS;
- submission authorization.

## Current execution decision

The frozen CUDA run remains authoritative and stays queued. CPU shadow work proceeds independently so free compute can eventually provide supporting evidence without consuming official OOF or Public LB.