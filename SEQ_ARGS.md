> **WARNING:** The Fpy specification is a work-in-progress

# Fpy Sequence Arguments and Sequence Calling

This document specifies how an Fpy sequence declares arguments, how argument values are bound when a sequence starts, and how a running sequence starts another sequence. It is a companion to the main [Fpy specification](SPEC.md) and follows its conventions. Terms such as [variable](SPEC.md#variables), [type](SPEC.md#types), [coercion](SPEC.md#type-conversion), [name group](SPEC.md#name-groups) and [command](SPEC.md#commands) are defined there.

The last section, [Design notes](#design-notes), is non-normative.

# Sequences and the environment

A **sequence** is a single Fpy program, compiled as a unit.

The **environment** is whatever starts a sequence running: a ground operator, an F-Prime component, or another sequence via a [sequence-run command](#sequence-run-commands).

Each sequence has an **argument specification**: an ordered list of name-and-[type](SPEC.md#types) pairs, declared by its [sequence statement](#the-sequence-statement). A sequence with no sequence statement, or with an empty one, has an empty argument specification.

The compiled form of a sequence records its argument specification as an ordered list of triples (name, fully-qualified type name, size), where **size** is the length in bytes of the binary form of the type.

> The compiled form is otherwise left unspecified. Recording the argument specification in the compiled sequence is what allows type-checked calls between separately compiled sequences.

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

*Tests:* [1](test/fpy/test_sequence_metadata.py#L301 "test/fpy/test_sequence_metadata.py::test_defining_sequence_in_function"), [2](test/fpy/test_sequence_metadata.py#L310 "test/fpy/test_sequence_metadata.py::test_defining_sequence_in_loop"), [3](test/fpy/test_sequence_metadata.py#L319 "test/fpy/test_sequence_metadata.py::test_defining_sequence_in_if_stmt")

If a sequence statement is present, it must be the first statement of the sequence.

> This implies there is at most one sequence statement per sequence.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L140 "test/fpy/test_sequence_metadata.py::test_sequence_after_statement"), [2](test/fpy/test_sequence_metadata.py#L105 "test/fpy/test_sequence_metadata.py::test_duplicate_sequence_statement"), [3](test/fpy/test_sequence_metadata.py#L24 "test/fpy/test_sequence_metadata.py::test_empty_sequence"), [4](test/fpy/test_sequence_metadata.py#L48 "test/fpy/test_sequence_metadata.py::test_sequence_with_trailing_comma")

Each `parameter_name` is resolved in the value name group. Each `parameter_type` is resolved in the type name group.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L132 "test/fpy/test_sequence_metadata.py::test_sequence_with_invalid_type"), [2](test/fpy/test_sequence_metadata.py#L227 "test/fpy/test_sequence_metadata.py::test_sequence_literal_as_type"), [3](test/fpy/test_sequence_metadata.py#L235 "test/fpy/test_sequence_metadata.py::test_sequence_bool_as_type")

### Semantics

Each parameter defines a variable named `parameter_name` with type `parameter_type` in the global scope, exactly as if by a [variable definition statement](SPEC.md#variable-definition), except:
* the variable is considered defined starting from the first statement of the sequence, and
* its initial value is supplied by the environment, per [argument binding](#argument-binding).

> In all other respects, sequence arguments are ordinary variables: they may be read, reassigned, passed to functions, and shadowed in inner scopes, and they occupy only the value name group, so they never conflict with types or callables.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L56 "test/fpy/test_sequence_metadata.py::test_sequence_parameter_as_variable"), [2](test/fpy/test_sequence_metadata.py#L556 "test/fpy/test_sequence_metadata.py::test_modify_arg"), [3](test/fpy/test_sequence_metadata.py#L170 "test/fpy/test_sequence_metadata.py::test_sequence_parameter_in_function"), [4](test/fpy/test_sequence_metadata.py#L269 "test/fpy/test_sequence_metadata.py::test_sequence_param_same_name_as_func"), [5](test/fpy/test_sequence_metadata.py#L280 "test/fpy/test_sequence_metadata.py::test_sequence_param_shadowed_by_loop_var"), [6](test/fpy/test_sequence_metadata.py#L290 "test/fpy/test_sequence_metadata.py::test_sequence_param_shadowed_by_func_param")

Because each parameter is a variable definition in the global scope, no two parameters may share a name, and no other global variable definition may share a name with a parameter.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L114 "test/fpy/test_sequence_metadata.py::test_duplicate_parameter_names"), [2](test/fpy/test_sequence_metadata.py#L122 "test/fpy/test_sequence_metadata.py::test_sequence_parameter_conflicts_with_variable")

If `parameter_type` is not [constant-sized](SPEC.md#types), an error is raised.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L243 "test/fpy/test_sequence_metadata.py::test_sequence_string_type_parameter"), [2](test/fpy/test_sequence_metadata.py#L251 "test/fpy/test_sequence_metadata.py::test_sequence_struct_with_string_member"), [3](test/fpy/test_sequence_metadata.py#L87 "test/fpy/test_sequence_metadata.py::test_sequence_with_struct_type"), [4](test/fpy/test_sequence_metadata.py#L96 "test/fpy/test_sequence_metadata.py::test_sequence_with_array_type"), [5](test/fpy/test_sequence_metadata.py#L183 "test/fpy/test_sequence_metadata.py::test_sequence_with_enum_type")

A sequence argument is not a [constant expression](SPEC.md#expressions).

> For example, a sequence argument cannot be the default value of a function parameter.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L259 "test/fpy/test_sequence_metadata.py::test_sequence_param_as_default_arg")

If a sequence statement has more than 255 parameters, an error is raised.

If a parameter's name, or the fully-qualified name of a parameter's type, is longer than 255 UTF-8 bytes, an error is raised.

> These limits let the compiled argument specification store the count in one byte and each name with a one-byte length prefix.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L584 "test/fpy/test_sequence_metadata.py::test_too_many_parameters"), [2](test/fpy/test_seq_calling.py#L649 "test/fpy/test_seq_calling.py::TestSeqArgLimits::test_arg_name_too_long"), [3](test/fpy/test_seq_calling.py#L658 "test/fpy/test_seq_calling.py::TestSeqArgLimits::test_arg_name_exactly_255_bytes")

The argument specification of the sequence is the ordered list of (`parameter_name`, `parameter_type`) pairs.

If the argument specification of a sequence is non-empty, that sequence cannot be [imported](SPEC.md#imports).

## Argument binding

To start a sequence, the environment supplies an **argument buffer**: the binary forms of one value per entry of the argument specification, in order, concatenated.

Before the first statement of the sequence executes:
1. If the length of the supplied argument buffer is not equal to the sum of the sizes of the argument specification, the sequence fails to start, and no statement executes.
2. Otherwise, each sequence argument's initial value is the value of its declared type whose binary form is the corresponding slice of the argument buffer.

> The buffer is validated only by total length. The environment is trusted to supply well-formed values of the declared types; there is no per-value check.

*Tests:* [1](test/fpy/test_sequence_metadata.py#L384 "test/fpy/test_sequence_metadata.py::test_arg_value_u32"), [2](test/fpy/test_sequence_metadata.py#L448 "test/fpy/test_sequence_metadata.py::test_multiple_args_correct_offsets"), [3](test/fpy/test_sequence_metadata.py#L354 "test/fpy/test_sequence_metadata.py::test_run_sequence_no_args_expected_none_provided"), [4](test/fpy/test_sequence_metadata.py#L363 "test/fpy/test_sequence_metadata.py::test_run_sequence_args_wrong_size"), [5](test/fpy/test_sequence_metadata.py#L373 "test/fpy/test_sequence_metadata.py::test_run_sequence_args_expected_but_missing")

# Sequence calling

A sequence starts another sequence by calling a sequence-run command. The called sequence is the **target**; its compiled form is the **target binary**.

## The Svc.SeqArgs type

The dictionary must define the type `Svc.SeqArgs` as a struct with exactly two members, in order:
1. `size`: an unsigned integer type
2. `buffer`: an array of `U8` with positive length

The **argument buffer capacity** is the length of `buffer`.

> `Svc.SeqArgs` carries an argument buffer inside a command. Its capacity is set per-deployment in the dictionary; the compiler adopts whatever capacity the dictionary declares.

*Tests:* [1](test/fpy/test_seq_calling.py#L700 "test/fpy/test_seq_calling.py::TestSeqArgsBufferSizeFromDictionary::test_non_default_buffer_size_loads_cleanly")

## Sequence-run commands

A **sequence-run command** is a command whose F-Prime parameters are, in order, exactly:
1. a string type
2. `Svc.BlockState`
3. `Svc.SeqArgs`

> In the reference FpySequencer this is the `RUN_ARGS` command, with parameters `fileName`, `block`, `buffer`. Any command matching the shape is treated as a sequence-run command; commands that do not match (such as `RUN` or `VALIDATE_ARGS`) are ordinary commands.

*Tests:* [1](test/fpy/test_seq_calling.py#L52 "test/fpy/test_seq_calling.py::TestSeqRunDetection::test_run_args_detected_as_seq_run"), [2](test/fpy/test_seq_calling.py#L70 "test/fpy/test_seq_calling.py::TestSeqRunDetection::test_regular_run_not_seq_run")

The Fpy [callable](SPEC.md#callables) corresponding to a sequence-run command does not take the `Svc.SeqArgs` parameter. Its parameters are the first two F-Prime parameters (name it `file_name` and `block`), followed by the parameters of the target's argument specification, in order and under their declared names.

> The caller passes the target's arguments directly, by position or by name, as if calling a function with the target's signature: `Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, x=42)`.

*Tests:* [1](test/fpy/test_seq_calling.py#L87 "test/fpy/test_seq_calling.py::TestSeqCallingNoArgs::test_call_child_no_args"), [2](test/fpy/test_seq_calling.py#L105 "test/fpy/test_seq_calling.py::TestSeqCallingWithArgs::test_call_child_one_u32_arg"), [3](test/fpy/test_seq_calling.py#L120 "test/fpy/test_seq_calling.py::TestSeqCallingWithArgs::test_call_child_multiple_args"), [4](test/fpy/test_seq_calling.py#L446 "test/fpy/test_seq_calling.py::TestSeqCallingNamedArgs::test_single_named_arg"), [5](test/fpy/test_seq_calling.py#L481 "test/fpy/test_seq_calling.py::TestSeqCallingNamedArgs::test_named_args_reordered"), [6](test/fpy/test_seq_calling.py#L499 "test/fpy/test_seq_calling.py::TestSeqCallingNamedArgs::test_mixed_positional_and_named")

## Target resolution

The `file_name` argument of a sequence-run command call must be a string literal; otherwise an error is raised. Its value is both the path at which the running F-Prime system will load the target binary, and the key by which the compiler locates a copy of the target binary.

The **ground binary directory** is a directory provided by the environment in which the compiler is invoked.

> In the command-line compiler, this is the `-g`/`--ground-binary-dir` option, defaulting to the directory containing the input sequence.

For each call to a sequence-run command:
1. If no ground binary directory was provided, an error is raised.
2. The `file_name` value is joined to the ground binary directory. If no file exists at the resulting path, an error is raised.
3. The argument specification recorded in that file is read. If the file cannot be read as a compiled sequence, an error is raised.
4. Each type name in the argument specification is resolved to a type. If a type name does not name a known type, or the recorded size differs from the size of the resolved type's binary form, an error is raised.

*Tests:* [1](test/fpy/test_seq_calling.py#L275 "test/fpy/test_seq_calling.py::TestSeqCallingErrors::test_missing_bin_file"), [2](test/fpy/test_seq_calling.py#L286 "test/fpy/test_seq_calling.py::TestSeqCallingErrors::test_no_binary_dir")

> The compiler checks the call against a ground copy of the target binary; the running system loads its own copy by the same name. Nothing verifies that the two copies are identical. If they disagree, the mismatch is caught at run time only if the total argument size differs (see [argument binding](#argument-binding)).

## Call checking

The call is checked as an ordinary [function call expression](SPEC.md#function-call-expression) against the parameter list defined in [sequence-run commands](#sequence-run-commands): arguments may be positional or named, every parameter must be supplied exactly once, and no unknown names may be supplied. Each argument for a target parameter must be [coercible](SPEC.md#type-conversion) to that parameter's declared type; otherwise an error is raised.

*Tests:* [1](test/fpy/test_seq_calling.py#L238 "test/fpy/test_seq_calling.py::TestSeqCallingErrors::test_wrong_arg_count"), [2](test/fpy/test_seq_calling.py#L258 "test/fpy/test_seq_calling.py::TestSeqCallingErrors::test_wrong_arg_type"), [3](test/fpy/test_seq_calling.py#L557 "test/fpy/test_seq_calling.py::TestSeqCallingNamedArgErrors::test_unknown_named_arg"), [4](test/fpy/test_seq_calling.py#L579 "test/fpy/test_seq_calling.py::TestSeqCallingNamedArgErrors::test_duplicate_named_arg"), [5](test/fpy/test_seq_calling.py#L601 "test/fpy/test_seq_calling.py::TestSeqCallingNamedArgErrors::test_positional_and_named_conflict"), [6](test/fpy/test_seq_calling.py#L623 "test/fpy/test_seq_calling.py::TestSeqCallingNamedArgErrors::test_missing_named_arg")

If the sum of the sizes of the target's argument specification exceeds the argument buffer capacity, an error is raised.

*Tests:* [1](test/fpy/test_seq_calling.py#L712 "test/fpy/test_seq_calling.py::TestSeqArgsBufferSizeFromDictionary::test_oversized_args_use_dictionary_capacity"), [2](test/fpy/test_seq_calling.py#L731 "test/fpy/test_seq_calling.py::TestSeqArgsBufferSizeFromDictionary::test_args_still_bounded_by_dictionary_capacity")

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

*Tests:* [1](test/fpy/test_seq_calling.py#L364 "test/fpy/test_seq_calling.py::TestSeqCallingReturnStatus::test_branch_on_success"), [2](test/fpy/test_seq_calling.py#L383 "test/fpy/test_seq_calling.py::TestSeqCallingReturnStatus::test_branch_on_child_failure"), [3](test/fpy/test_seq_calling.py#L214 "test/fpy/test_seq_calling.py::TestSeqCallingWithArgs::test_wrong_value_causes_failure")

A target may itself call sequence-run commands, to any depth supported by the running system.

*Tests:* [1](test/fpy/test_seq_calling.py#L307 "test/fpy/test_seq_calling.py::TestSeqCallingNested::test_nested_two_levels"), [2](test/fpy/test_seq_calling.py#L335 "test/fpy/test_seq_calling.py::TestSeqCallingNested::test_nested_pass_through_arg")

# Design notes

This section is non-normative. It records trade-offs in the current design and possible improvements. The original design rationale is in [issue #39](https://github.com/fprime-community/fpy/issues/39); the notes below stay within the decisions made there.

**The call syntax is settled.** Issue #39 chose command-call syntax over the alternatives. Import-style calling (`import some_seq from "path.bin"` then `some_seq(1, 2, 3)`) was the initial choice but was rejected: it costs two lines per call, gives no way to name the sequencer instance that runs the target (the command receiver expresses this for free), and borrows a Python mental model for something fundamentally different. That last point is stronger now than when it was written: Fpy `import` means compile-time inlining, while a sequence call is runtime dispatch to a separately compiled program. A custom statement (`run "seq.bin" no_block args(1, 2)`) was rejected as new syntax for operators to learn. The notes below therefore keep the command-call design and address its edges.

**Structural command detection is fragile.** A sequence-run command is recognized purely by its parameter shape (string, `Svc.BlockState`, `Svc.SeqArgs`). This keeps argument types and counts out of the flight software, which is a requirement of #39, but shape-matching is a heuristic that can misfire in both directions: an unrelated command that happens to match the shape gets its signature silently rewritten, and a sequencer command that renames or reorders parameters silently loses checking. An explicit designation -- an FPP annotation on the command, or a compiler configuration entry naming the commands -- would keep the FSW-independence of the argument types while making the treatment intentional.

**The file name does double duty.** One string literal is both the onboard load path and the ground lookup key, forcing the ground directory layout to mirror the onboard one. Without changing the call syntax, the compiler could resolve the ground copy through a search path or a configurable path mapping (as imports resolve sources) while still recording the onboard path verbatim in the command.

**Interface identity is name-plus-size only.** Target resolution accepts a type if its name and byte size match. Two dictionary versions can disagree about a struct's field layout while agreeing on both, and the call still compiles. #39 deliberately limited runtime validation to total-size matching to avoid fragile string comparisons; recording a structural hash of each type in the argument specification would strengthen the compile-time check, and would give the sequencer a fixed-width comparison to perform onboard -- satisfying the issue's runtime-type-validation nice-to-have without comparing strings.

**No staleness protection.** The compiler checks the ground copy of the target binary, but nothing ties the call to the onboard copy; only the total argument size is validated at run time. Recording the target binary's CRC in the caller and having the sequencer compare it (perhaps optionally) would close this gap, again as a fixed-width comparison in the same spirit as the size check.

**`Svc.SeqArgs` is always full capacity.** The struct serializes at fixed size, so every sequence-run command carries the whole buffer, padding included, regardless of how many argument bytes are used. This bloats uplinked commands and couples the max argument size to the command buffer size. A variable-length encoding would remove the waste, at the cost of departing from plain FPP struct serialization.

**No default values.** Function parameters may have constant defaults, but sequence parameters may not. Since the argument specification already carries per-argument metadata, constant defaults could be recorded there and filled in by callers that omit the argument.

**`RUN_ARGS` is a stopgap.** #39 describes the separate `RUN_ARGS` opcode as a temporary workaround for GDS compatibility, so `RUN` and `RUN_ARGS` may eventually merge. Until then the split has rough edges: `RUN` on an argument-taking sequence fails only at run time (size mismatch), and `VALIDATE_ARGS` does not match the sequence-run shape, so calling it from Fpy requires constructing a raw `Svc.SeqArgs`, which is impractical. Explicit designation (above) would let `VALIDATE_ARGS` receive the same vararg treatment, and the compiler could warn when `RUN` targets a sequence with a non-empty argument specification.
