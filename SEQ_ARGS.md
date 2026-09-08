> **WARNING:** The Fpy specification is a work-in-progress

# Fpy Sequence Arguments and Sequence Calling

This document specifies how an Fpy sequence declares arguments, how argument values are bound when a sequence starts, and how a running sequence starts another sequence. It is a companion to the main [Fpy specification](SPEC.md) and follows its conventions. Terms such as [variable](SPEC.md#variables), [type](SPEC.md#types), [coercion](SPEC.md#type-conversion), [name group](SPEC.md#name-groups) and [command](SPEC.md#commands) are defined there.

The last section, [Design notes](#design-notes), is non-normative.

# Sequences and the environment

A **sequence** is a single Fpy program, compiled as a unit.

The **environment** is whatever starts a sequence running: a ground operator, an F-Prime component, or another sequence via a [sequence-run command](#sequence-run-commands).

Each sequence has an **argument specification**: an ordered list of name-and-[type](SPEC.md#types) pairs, declared by its [sequence statement](#the-sequence-statement). A sequence with no sequence statement, or with an empty one, has an empty argument specification.

The compiled form of a sequence records its argument specification as an ordered list of triples (name, fully-qualified type name, size), where **size** is the length in bytes of the binary form of the type.

> The compiled form is otherwise left unspecified. The recorded argument specification is what the running system uses for [argument binding](#argument-binding); the compiler checks calls between separately compiled sequences against the target's source instead (see [target resolution](#target-resolution)).

# Sequence arguments

A **sequence argument** is a [variable](SPEC.md#variables) implicitly defined by the sequence statement, whose initial value is supplied by the environment when the sequence starts.

## The sequence statement

A **sequence statement** declares the argument specification of the sequence.

### Syntax

Rule:

```
sequence_stmt: "sequence" "(" [sequence_parameters] ")"
sequence_parameters: sequence_parameter ("," sequence_parameter)* [","]
sequence_parameter: name ":" qualified_name
```

Name:

```
sequence_stmt: "sequence" "(" parameters ")"
sequence_parameter: parameter_name ":" parameter_type
```

A sequence statement is only valid outside an indentation block.

If a sequence statement is present, it must be the first statement of the sequence.

> This implies there is at most one sequence statement per sequence.

Each `parameter_name` is resolved in the value name group. Each `parameter_type` is resolved in the type name group.

### Semantics

Each parameter defines a variable named `parameter_name` with type `parameter_type` in the global scope, exactly as if by a [variable definition statement](SPEC.md#variable-definition), except:
* the variable is considered defined starting from the first statement of the sequence, and
* its initial value is supplied by the environment, per [argument binding](#argument-binding).

> In all other respects, sequence arguments are ordinary variables: they may be read, reassigned, passed to functions, and shadowed in inner scopes, and they occupy only the value name group, so they never conflict with types or callables.

Because each parameter is a variable definition in the global scope, no two parameters may share a name, and no other global variable definition may share a name with a parameter.

If `parameter_type` is not [constant-sized](SPEC.md#types), an error is raised.

A sequence argument is not a [constant expression](SPEC.md#expressions).

> For example, a sequence argument cannot be the default value of a function parameter.

If a sequence statement has more than 255 parameters, an error is raised.

If a parameter's name, or the fully-qualified name of a parameter's type, is longer than 255 UTF-8 bytes, an error is raised.

> These limits let the compiled argument specification store the count in one byte and each name with a one-byte length prefix.

The argument specification of the sequence is the ordered list of (`parameter_name`, `parameter_type`) pairs.

If the argument specification of a sequence is non-empty, that sequence cannot be [imported](SPEC.md#imports).

## Argument binding

To start a sequence, the environment supplies an **argument buffer**: the binary forms of one value per entry of the argument specification, in order, concatenated.

Before the first statement of the sequence executes:
1. If the length of the supplied argument buffer is not equal to the sum of the sizes of the argument specification, the sequence fails to start, and no statement executes.
2. Otherwise, each sequence argument's initial value is the value of its declared type whose binary form is the corresponding slice of the argument buffer.

> The buffer is validated only by total length. The environment is trusted to supply well-formed values of the declared types; there is no per-value check.

# Sequence calling

A sequence starts another sequence by calling a sequence-run command. The called sequence is the **target**; its compiled form is the **target binary**.

## The Svc.SeqArgs type

The dictionary must define the type `Svc.SeqArgs` as a struct with exactly two members, in order:
1. `size`: an unsigned integer type
2. `buffer`: an array of `U8` with positive length

The **argument buffer capacity** is the length of `buffer`.

> `Svc.SeqArgs` carries an argument buffer inside a command. Its capacity is set per-deployment in the dictionary; the compiler adopts whatever capacity the dictionary declares.

## Sequence-run commands

A **sequence-run command** is a command whose F-Prime parameters are, in order, exactly:
1. a string type
2. `Svc.BlockState`
3. `Svc.SeqArgs`

> In the reference FpySequencer this is the `RUN_ARGS` command, with parameters `fileName`, `block`, `buffer`. Any command matching the shape is treated as a sequence-run command; commands that do not match (such as `RUN` or `VALIDATE_ARGS`) are ordinary commands.

The Fpy [callable](SPEC.md#callables) corresponding to a sequence-run command does not take the `Svc.SeqArgs` parameter. Its parameters are the first two F-Prime parameters (name it `file_name` and `block`), followed by the parameters of the target's argument specification, in order and under their declared names.

> The caller passes the target's arguments directly, by position or by name, as if calling a function with the target's signature: `Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, x=42)`.

## Target resolution

The `file_name` argument of a sequence-run command call must be a string literal; otherwise an error is raised. Its value is both the path at which the running F-Prime system will load the target binary, and the key by which the compiler locates the target's source.

The **sequence path mappings** are an ordered list of (binary prefix, source prefix) pairs provided by the environment in which the compiler is invoked.

> In the command-line compiler, this is the `--seq-map BIN_PREFIX=FPY_PREFIX` option, repeatable, with no default. An empty binary prefix matches every path.

For each call to a sequence-run command:
1. If no sequence path mappings were provided, an error is raised.
2. For each mapping in order whose binary prefix is a prefix of the `file_name` value, a candidate path is formed by replacing that prefix with the source prefix and the file name's extension with `.fpy`. The **target source** is the first candidate at which a file exists. If there is none, an error is raised.
3. The target source is lexed and parsed. If this fails, or its sequence statement is not its first statement, an error is raised.
4. The argument specification is read from the target's sequence statement. Each `parameter_type` name is resolved as a primitive type name or a fully-qualified dictionary type name; if it is neither, an error is raised.

> Only the target's sequence statement is read. Its other statements, including its imports, are neither resolved nor checked; the target is fully checked when it is compiled itself.

> The compiler checks the call against the target's source; the running system loads the compiled binary by the `file_name` value. Nothing verifies that the binary was built from that source. If they disagree, the mismatch is caught at run time only if the total argument size differs (see [argument binding](#argument-binding)).

## Call checking

The call is checked as an ordinary [function call expression](SPEC.md#function-call-expression) against the parameter list defined in [sequence-run commands](#sequence-run-commands): arguments may be positional or named, every parameter must be supplied exactly once, and no unknown names may be supplied. Each argument for a target parameter must be [coercible](SPEC.md#type-conversion) to that parameter's declared type; otherwise an error is raised.

If the sum of the sizes of the target's argument specification exceeds the argument buffer capacity, an error is raised.

## Evaluation

A sequence-run command call is evaluated per [command evaluation](SPEC.md#command-evaluation), with the underlying F-Prime command's third argument constructed as the `Svc.SeqArgs` value whose:
* `size` is the sum of the sizes of the target's argument specification, and
* `buffer` is the argument values, coerced to their declared types and serialized in argument specification order, followed by zero bytes up to the argument buffer capacity.

> The target receives exactly the argument buffer described in [argument binding](#argument-binding); the zero padding is not part of it.

The target executes as its own program: it does not share variables, functions, or any other state with the caller, and the caller observes only the command response.

The expression evaluates to the command response:
* If `block` is `Svc.BlockState.BLOCK`, the response arrives when the target finishes: `Fw.CmdResponse.OK` if it ran to completion successfully, `Fw.CmdResponse.EXECUTION_ERROR` if it failed to load, failed [argument binding](#argument-binding), or halted with an error.
* If `block` is `Svc.BlockState.NO_BLOCK`, the response is `Fw.CmdResponse.OK` as soon as the command is accepted, before the target's outcome is known.
* In either case, the response is `Fw.CmdResponse.EXECUTION_ERROR` if the command cannot be accepted (for example, the receiving sequencer is busy).

> Because command evaluation blocks until the response arrives, `BLOCK` runs the target synchronously, and `NO_BLOCK` runs it concurrently with the rest of the calling sequence.

> Per the semantics of bare command calls, a `BLOCK` call whose response is discarded halts the calling sequence if the target fails, unless `flags.assert_cmd_success` is `False`. Saving the response in a variable, or using the response in any way, suppresses this.

A target may itself call sequence-run commands, to any depth supported by the running system.

# Design notes

This section is non-normative. It records trade-offs in the current design and possible improvements. The original design rationale is in [issue #39](https://github.com/fprime-community/fpy/issues/39); the notes below stay within the decisions made there.

**The call syntax is settled.** Issue #39 chose command-call syntax over the alternatives. Import-style calling (`import some_seq from "path.bin"` then `some_seq(1, 2, 3)`) was the initial choice but was rejected: it costs two lines per call, gives no way to name the sequencer instance that runs the target (the command receiver expresses this for free), and borrows a Python mental model for something fundamentally different. That last point is stronger now than when it was written: Fpy `import` means compile-time inlining, while a sequence call is runtime dispatch to a separately compiled program. A custom statement (`run "seq.bin" no_block args(1, 2)`) was rejected as new syntax for operators to learn. The notes below therefore keep the command-call design and address its edges.

**Structural command detection is fragile.** A sequence-run command is recognized purely by its parameter shape (string, `Svc.BlockState`, `Svc.SeqArgs`). This keeps argument types and counts out of the flight software, which is a requirement of #39, but shape-matching is a heuristic that can misfire in both directions: an unrelated command that happens to match the shape gets its signature silently rewritten, and a sequencer command that renames or reorders parameters silently loses checking. An explicit designation -- an FPP annotation on the command, or a compiler configuration entry naming the commands -- would keep the FSW-independence of the argument types while making the treatment intentional.

**The file name does double duty.** One string literal is both the onboard load path and the ground lookup key. The sequence path mappings decouple the two layouts: the compiler resolves the target's source through them while still recording the onboard path verbatim in the command.

**Interface identity is checked against the source, not the binary.** The compiler resolves the target's parameter types from its source under the calling compilation's dictionary; the argument specification recorded in the target's compiled binary is never compared against it. Two dictionary versions can disagree about a type's layout and the call still compiles. #39 deliberately limited runtime validation to total-size matching to avoid fragile string comparisons; recording a structural hash of each type in the argument specification would give the sequencer a fixed-width comparison to perform onboard -- satisfying the issue's runtime-type-validation nice-to-have without comparing strings.

**No staleness protection.** The compiler checks the target's source, but nothing ties the call to the onboard binary: neither that the binary was built from that source, nor that it is current. Only the total argument size is validated at run time. Recording the target binary's CRC in the caller and having the sequencer compare it (perhaps optionally) would close this gap, again as a fixed-width comparison in the same spirit as the size check.

**`Svc.SeqArgs` is always full capacity.** The struct serializes at fixed size, so every sequence-run command carries the whole buffer, padding included, regardless of how many argument bytes are used. This bloats uplinked commands and couples the max argument size to the command buffer size. A variable-length encoding would remove the waste, at the cost of departing from plain FPP struct serialization.

**No default values.** Function parameters may have constant defaults, but sequence parameters may not. Since the argument specification already carries per-argument metadata, constant defaults could be recorded there and filled in by callers that omit the argument.

**`RUN_ARGS` is a stopgap.** #39 describes the separate `RUN_ARGS` opcode as a temporary workaround for GDS compatibility, so `RUN` and `RUN_ARGS` may eventually merge. Until then the split has rough edges: `RUN` on an argument-taking sequence fails only at run time (size mismatch), and `VALIDATE_ARGS` does not match the sequence-run shape, so calling it from Fpy requires constructing a raw `Svc.SeqArgs`, which is impractical. Explicit designation (above) would let `VALIDATE_ARGS` receive the same vararg treatment, and the compiler could warn when `RUN` targets a sequence with a non-empty argument specification.
