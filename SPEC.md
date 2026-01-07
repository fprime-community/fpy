> **WARNING:** The Fpy specification is a work-in-progress

# Fpy Specification

Fpy is a sequencing language for the F-Prime flight software framework. It combines Python-like syntax with the FPP model, with domain-specific features for spacecraft operation.

This document specifies the syntax and semantics of the Fpy sequencing language. In other words, it specifies which programs the compiler should accept, and how the output should behave when run. The intent is to leave the implementation of the compiler, the bytecode it generates, and the virtual machine it runs on, unspecified. 

It is assumed the reader is familiar with the FPP model, including commands, telemetry, parameters, structs, arrays and enums.

> Informal notes and explanations are quoted like this.

Terms are **bolded** in the location they are primarily defined.

`Monospaced` text refer to syntactic rules, type names or example Fpy code.

In a syntactic rule:
* Text in between forward slashes `/` is regex
* Text in between square brackets `[]` is optional
* Text in between parentheses `()` is handled as a group
* A plus suffix `+` means one or more instances of its preceding rule
* A star suffix `*` means zero or more instances of its preceding rule
* A question mark suffix `?` means zero or one instances of its preceding rule

# Names and scopes

## Names

`name: /\$?[^\W\d]\w*/`

A **name** is a string consisting of letters, underscores or digits. The first character may not be a digit. 

The first character may optionally be `$`, in which case the name is considered **escaped**. An escaped name is the same as an unescaped name, except in that it always lexes as a name, even if it is a [reserved word](#reserved-words).

## Reserved words

A **reserved word** is a word which cannot be used as an [unescaped name](#names).

The list of reserved words is:
* `assert`
* `break`
* `check`
* `continue`
* `def`
* `for`
* `if`
* `not`
* `pass`
* `return`
* `while`

## Symbols

A **symbol** is a language construct that can be referred to by a name in the program. 

The following language constructs may be symbols:
* [namespaces](#namespaces)
* [variables](#variables)
* [callables](#callables)
* [types](#types)
* telemetry channels
* parameters
* enum constants
TODO members?
TODO you're selecting a member of a value--not referring to a static member. when you say a.b you're asking fgor a comp to be performed

## Scopes

A **scope** is a mapping of names to symbols, accessed via some region of the source code.
TODO a scope IS a REGION

The **global scope** is the scope accessible throughout the entire source code.

Each [function](#functions) has a **function scope**, accessible in its body.

The **resolving scope** is the most specific scope that some part of the source code has access to.

Scopes may have a **parent scope**:
* The parent scope of a function scope is the global scope.
* The global scope does not have a parent scope.

## Name groups

Scopes are divided into **name groups**.

The list of name groups is:
* The **[value](#values) name group**
* The **[type](#types) name group**
* The **[callable](#callables) name group**

Each name group only contains names which map to their particular language construct, or namespaces with the same property, recursively.
TODO Maybe could remove this line

TODO explain why we need this
Name groups do not intersect.

> This means that the names of callables, types and values never conflict.  WITH EACH OTHER

Name groups are accessed via syntactic context.
TODO this is really just an ecplanation

TODO A.b is an expr, A is an expr, b is not an expression. A.b is a dot expression. b is not an expression, it's part of a 
TODO Does it refer to something that has a scope, or does it refer to something that has members.

> For instance, the type name group is accessible anywhere in the source code where a type name is expected, such as a [variable definition](#variable-definition) type annotation, or a [function definition](#function-definition) return type.

The **resolving name group** is the name group that a name should be resolved in, based on its syntactic context.

## Namespaces

A **namespace** is a mapping of names to symbols, associated with a name.

## Qualified names

A **qualified name** is one of:
* A name
* A qualified name, followed by a `.`, followed by a name

The **qualifier** is the qualified name to the left of the `.`.

To resolve a qualified name in a name group:
1. If there is no qualifier:
    1. Resolve the name in the resolving scope.
    2. If the name fails to be resolved, resolve the name in the parent scope of the resolving scope.
2. Otherwise:
    1. Resolve the qualifier.
    2. If the qualifier is an expression, resolution is handled by the rules of [member access](#member-access-expression).
    3. If the qualifier is not a namespace, an error is raised.
    4. Resolve the name in the qualifier namespace.

If at any point a name fails to be resolved, an error is raised, unless otherwise specified.

A **fully-qualified name** is a qualified name which is not itself a qualifier.

If a fully-qualified name resolves to a namespace, an error is raised.

> Namespace symbols cannot be used anywhere, so this forces names to resolve to something "useful"

TODO I'm not sure this is clear what this means. The idea here is that the full qualified name should always reference SOMETHING--cannot just put Svc in place of a type, even though Svc does resolve in type name group and global scope.

## Definitions

A **definition** is a language construct that introduces a name-to-[symbol](#symbols) mapping as apart ofa  a [scope](#scopes) and [name group](#name-groups).

The list of definitions is:
* [Variable definitions](#variable-definition)
* [Function definitions](#function-definition)

# Variables

A **variable** is a symbol with a static [type](#types) and a dynamic, mutable [value](#values).
todo maybe call it dynamic?

TODO: give a list of statements

## Variable definition

A **variable definition statement** introduces a name-to-variable mapping in its resolving scope.

### Syntax

Rule:

`variable_declare_stmt: name ":" qualified_name "=" expr`

Name:

`variable_declare_stmt: lhs ":" type_ann "=" rhs`

`lhs` and `rhs` are resolved in the value name group.

`type_ann` is resolved in the type name group.

### Semantics

If `lhs` resolves to a previously-defined symbol, an error is raised.

> This prevents redefining a variable.

If `rhs` cannot be coerced to type `type_ann`, an error is raised.

The new variable has type `type_ann`. It is added to the resolving scope under name `lhs`.

At execution, `rhs` is [evaluated](#evaluation) and [coerced](#type-coercion) to type `type_ann`. This becomes the variable's initial value.

For statements following this, the variable `lhs` is considered defined.

## Variable assignment

A **variable assignment statement** mutates the value of a variable.

### Syntax
Rule:

`variable_assign_stmt: name "=" expr`

Name:

`variable_assign_stmt: lhs "=" rhs`

`lhs` and `rhs` are resolved in the value name group.

### Semantics

If `lhs` does not resolve to a variable, an error is raised.

> Hereafter, `lhs` refers to the variable named `lhs`.

If `lhs` has not been defined yet, an error is raised.

If `rhs` cannot be coerced to `lhs`'s type, an error is raised.

At execution, `rhs` is [evaluated](#evaluation) and [coerced](#type-coercion) to `lhs`'s type. This becomes the `lhs`'s new value.

## Member assignment

A **member assignment statement** mutates the value of a [member](#structs) in a variable.

### Syntax
Rule:

`variable_assign_member_stmt: expr "." name "=" expr`

Name:

`variable_assign_member_stmt: parent "." member "=" rhs`

`parent`, and `rhs` are resolved in the value name group.

### Semantics

If `parent` is not a variable or a [field](#fields) with a [field base](#fields) that is a variable, an error is raised.

> This allows for setting a field of a field to arbitrary depth, as long as the underlying thing you're modifying is a variable.

If `parent`'s type is:
* not [constant-sized](#types), or
* not a [struct](#structs) type, or
* `member` is not a member of `parent`'s type

... an error is raised.

If the variable has not been defined yet, an error is raised.

If `rhs` cannot be coerced to the member's type, an error is raised.

At execution, `rhs` is [evaluated](#evaluation) and [coerced](#type-coercion) to the member's type. This becomes the member's new value.

The value of the variable is unchanged except for the member.

## Element assignment

An **element assignment statement** mutates the value of an [element](todo) in a variable.

### Syntax
Rule:

`variable_assign_element_stmt: expr "[" expr "]" "=" expr`

Name:

`variable_assign_element_stmt: parent "[" item "]" "=" rhs`

`parent`, `item` and `rhs` are resolved in the value name group.

### Semantics

If `parent` is not a variable or a [field](#fields) with a [field base](todo) that is a variable, an error is raised.

> This allows for setting a field of a field to arbitrary depth, as long as the underlying thing you're modifying is a variable.

If the variable has not been defined yet, an error is raised.

If `item` cannot be [coerced](#type-coercion) to [array index type](#type-aliases), an error is raised.

If `parent`'s type is:
* not [constant-sized](#types), or
* not an [array](#arrays) type, or
* `item` is a [constant](todo) with a value less than 0 or greater than the `parent`'s type length, 

... an error is raised.

If `rhs` cannot be coerced to the element's type, an error is raised.

At execution:
1. `rhs` is evaluated and coerced to the element's type
2. `item` is [evaluated](#evaluation) and coerced to [array index type](#type-aliases)
3. If `item` is less than zero or greater than the `parent`'s type length, a runtime error is raised
4. The element in the `parent` array at the index `item` is set to the result of step 1

The value of the variable is unchanged except for the element.

## Variable evaluation

The value produced by [evaluating](#expressions) a variable is the value most recently assigned to that variable, or the initial value if it has only been defined.

If a variable is evaluated before it has been defined, an error is raised.

# Functions

A **function** is a [callable](#callables) [symbol](#symbols) with an inner scope, parameters, code and a return [type](#types).

The **call site** is the location in the source code at which a function is called.

## Function parameters

A **function parameter** is a [variable](#variables) implicitly defined by a function in that function's scope.

When a function is [called](todo), each parameter is set to an initial value.

The initial value may either be from a passed [argument](todo), or a default value, if one is specified in the function definition.

> In all other respects, parameters are like normal variables, meaning you can modify them in the function body.

## Return types

The **return type** of a function is the [type](#types) of the value returned by that function. If the return type is [Nothing](#internal-types), the function does not return a value.

## Returns

A **return statement** ends the currently executing function, resumes execution at the call site, and optionally returns a value.

### Syntax
Rule:

`return_stmt: "return" [expr]`

Name:

`return_stmt: "return" value`

`value` is resolved in the value name group.

### Semantics

If the return statement is outside of a function body, an error is raised.

The **enclosing function** of a return statement is the function whose body that return is in.

If `value` is not provided and the enclosing function's return type is not Nothing, an error is raised.

If `value` is provided and cannot be [coerced](#type-coercion) to the return type of the enclosing function, an error is raised.

At execution:
1. If provided, `value` is [evaluated](todo) and [coerced](#type-coercion) to the return type of the enclosing function.
2. The execution of the function body is stopped, and execution after the function call site resumes.

## Function definition

A **function definition statement** introduces a name-to-[function](#function) mapping in the global scope.

### Syntax

Rule:

```
function_def_stmt: "def" name "(" [parameters] ")" ["->" qualified_name] ":" block
parameters: parameter ("," parameter)*
parameter: name ":" qualified_name ["=" expr]
```

Name:

```
function_def_stmt: "def" name "(" parameters ")" "->" return_type ":" body
parameters: parameter_0 "," parameter_1 ... "," parameter_n
parameter: parameter_name ":" parameter_type "=" parameter_default_value
```

A function definition statement is only valid outside an indentation block.

The parameter `name`s are resolved in the value name group.

`name` is resolved in the callable name group.

`return_type` and each of the parameter types are resolved in the type name group.

### Semantics

If `name` resolves to a previously-defined callable, an error is raised.

A new function [scope](#scopes) is created, accessible to the `body` and the parameter `name`s.

Each parameter is a variable in this new scope.

> This implies that no two parameters may have the same name, otherwise they would be conflicting variables.

If the default value of a parameter is not a [constant](todo), an error is raised.

If the default value of a parameter cannot be [coerced](#type-coercion) to the type of the parameter, an error is raised.

If a parameter without a default value follows a parameter with a default value, an error is raised.

If `return_type` is provided, and any [branch](todo) of the function does not return a value, an error is raised.
TODO need a section on control flow?

The new function with name `name` is added to the global scope. If `return_type` is not provided, the [return type](#return-types) is [Nothing](#internal-types), otherwise the return type is type `return_type`.

> Because functions can only be defined in the global scope, you cannot declare a function in a function.

> Functions can be used before they are defined.

# Ifs
An **if statement** conditionally executes blocks of code.

## Syntax
Rule:

`if_stmt: "if" expr ":" stmt_list elifs ["else" ":" stmt_list]`

`elifs: elif_*`

`elif_: "elif" expr ":" stmt_list`

Name:

`if_stmt: "if" if_condition ":" body elifs "else" ":" else_body`

`elif_: "elif" elif_condition ":" elif_body`

`if_condition` and all `elif_condition`s are resolved in the value name group.

## Semantics

If `if_condition` or any `elif_condition` cannot be [coerced](#type-coercion) to [`bool`](#boolean-type), an error is raised.

At execution, the conditions will be evaluated one at a time until one evaluates to `True`, starting from `if_condition` and going in order through the `elif_conditions`.

The body of the first condition to evaluate to `True` is executed, and then execution continues after the if statement.

If no condition evaluates to `True`, and an `else_body` was provided, that body is executed, and then execution continues after the if statement.

# Loops

A **loop** executes a block of code zero or more times.

Each loop has a **loop condition**, which is a Boolean expression which, when `True`, allows the loop body to execute.

The **enclosing loop** is the loop whose body some source code is in.

The list of loops is:
* [While loops](#while-loop-statement)
* [For loops](#for-loops)

## While loop statement

A **while statement** executes a block of code in a loop while a condition holds `True`.

### Syntax

Rule:

`while_stmt: "while" expr ":" stmt_list`

Name:

`while_stmt: "while" condition ":" body`

`condition` is resolved in the value name group.

### Semantics

If `condition` cannot be [coerced](#type-coercion) to [`bool`](#boolean-type), an error is raised.

The loop condition of a while loop is the provided `condition`.

At execution:
1. The loop condition is evaluated.
2. If the loop condition is `True`, execute the body, and return to step 1.
3. Otherwise, execution continues after the while loop statement.

## For loop statement

A **for loop statement** executes a block of code until a counter reaches an upper bound.

### Syntax
Rule:

`for_stmt: "for" name "in" expr ":" stmt_list`

Name:

`for_stmt: "for" loop_var "in" range ":" body`

`loop_var` and `range` are resolved in the value name group.

### Semantics

The **loop variable** of a for loop is the variable named by `loop_var`.

If `loop_var` resolves to a previously-defined variable:
1. If the type of that variable is not [loop var type](#type-aliases), an error is raised.
2. Otherwise, that variable becomes the loop variable of this for loop.

> This allows reusing the same loop variable name across multiple for loops.

If `loop_var` does not resolve to a previously-defined variable, a new variable with name `loop_var` and [loop var type](#type-aliases) is added to the [resolving scope](#scopes), and it becomes the loop variable of this for loop.

> Nothing prevents you from modifying the loop variable in the loop body. However, this may cause infinite loops, so do this with caution.

If `range` cannot be [coerced](#type-coercion) to [Range type](#internal-types), an error is raised.

The loop condition of a for loop is `loop_var < upper_bound`, where `upper_bound` is the upper bound of the `range` expression.

At execution:
1. `range` is evaluated.
2. The loop variable is set to the lower bound of `range`.
3. The loop condition is evaluated.
4. If the loop condition is `True`, execute the body, increment the value of the `loop_var` by 1, and return to step 1.
5. Otherwise, execution continues after the for loop statement.

> The only possible step size is 1.

## Break statement

A **break statement** stops execution of a loop.

### Syntax
Rule:

`break_stmt: "break"`

### Semantics

If the break statement is outside of a loop body, an error is raised.

At execution, the enclosing loop body stops executing, and execution is continued after the enclosing loop.

## Continue statement

A **continue statement** immediately skips the rest of the loop body, continuing on to the next iteration or ending the loop.

### Syntax
Rule:

`continue_stmt: "continue"`

### Semantics

If the continue statement is outside of a loop body, an error is raised.

At execution:
1. The enclosing loop body stops executing.
2. If the loop is a [for loop](#for-loop-statement), the loop variable is incremented.
3. The loop condition is re-evaluated.
4. The loop continues as specified based on the result of the loop condition, either running the body or ending the loop.

# Assert statement

An **assert statement** evaluates a Boolean expression and halts the program if the expression evaluates to `False`.

## Syntax
Rule:

`assert_stmt: "assert" expr ["," expr]`

Name:

`assert_stmt: "assert" condition "," exit_code`

`condition` and `exit_code` are resolved in the value name group.

## Semantics

If `condition` cannot be coerced to [`bool`](#boolean-type), an error is raised.

If `exit_code` is provided, and cannot be coerced to [`U8`](#primitive-numeric-types), an error is raised.

At execution, if `condition` evaluates to `False`:
1. If `exit_code` is provided, evaluate it and display its value to the user.
2. If `exit_code` is not provided, display a generic error code to the user.
3. Halt the program.

# Check statement
The **check statement** executes a block of code if a Boolean expression evaluates to `True` for a duration of time, checking with a configurable frequency and timing out at a configurable time.

## Syntax
Rule:

`check_stmt: "check" expr ["timeout" expr] ["persist" expr] ["freq" expr] ":" stmt_list ["timeout" ":" stmt_list]`

Name:

`check_stmt: "check" condition "timeout" timeout "persist" persist "freq" freq ":" body "timeout" ":" timeout_body`

`condition`, `timeout`, `persist`, and `freq` are resolved in the value name group.

## Semantics

If `condition` cannot be [coerced](#type-coercion) to [`bool`](#boolean-type), an error is raised.

If `timeout` is provided, and cannot be coerced to [`Fw.Time`](todo), an error is raised.

If `persist` or `freq` is provided, and they cannot be coerced to [`Fw.TimeIntervalValue`](todo), an error is raised.

At execution:
1. If provided, `timeout`, `persist` and `freq` are evaluated and stored.
2. If `persist` is not provided, its stored value is a zero-duration `Fw.TimeIntervalValue`.
3. If `freq` is not provided, its stored value is a one-second `Fw.TimeIntervalValue`.
4. If `timeout` was provided and the current time is [greater](todo) than `timeout`'s stored value, the check times out.
5. Evaluate `condition`.
6. If `condition` has evaluated to `True` for duration greater than or equal to `persist`'s stored value, execute `body`, then continue execution after the check statement.
7. Otherwise, [sleep](todo) for `freq`'s stored duration.
8. Go to step 4.

If the check times out during execution:
1. If `timeout_body` is provided, execute it.
2. Execution continues after the check statement.

> Not providing `persist`, or providing a zero-duration `persist`, means the `condition` only needs to evaluate to `True` once.
> The timeout defaults to never, and the frequency defaults to once per second.

If at any point during execution, two times which are [incomparable](todo) are attempted to be compared, the check statement will halt the program as if by an [assertion](#assert-statement), and display an error code.

# Callables

A **callable** is a symbol with parameters and a return [type](#types) which can be evaluated by being called.

> Evaluation of a callable always refers to evaluation of a [function call expression](#function-call-expression), where the function is that callable.
> Because callable evaluation is always via a function call expression, when talking about the evaluation of a callable, it is assumed that the evaluation semantics of the function call expression have already occurred. Specifically, the arguments have already been evaluated from left to right.

Callables can be divided into four categories:
* [Commands](#commands)
* [Functions](#functions)
* [Builtin functions](#builtin-functions)
* [Constructors](todo)

## Commands
A **command** is a callable with an associated F-Prime command.

All F-Prime commands in the dictionary have a corresponding Fpy command.

All Fpy commands are [globally-scoped](#scopes).

The fully-qualified name of an Fpy command is the same as the corresponding F-Prime command.

The parameter names and types of an Fpy command are the same as the corresponding F-Prime command.

> The F-Prime specification requires that all command parameter types be [serializable](#types).

The return type of all commands is [`Fw.CmdResponse`](todo).

> Throughout the specification, a "command" means an Fpy command.

### Command evaluation

Command evaluation is performed as follows:
1. The argument values are serialized and arranged into the F-Prime command binary format.
2. The binary command is dispatched to the F-Prime system.
3. Execution blocks until the [command response](todo) comes back from the F-Prime system.
4. The expression evaluates to a value of `Fw.CmdResponse` corresponding to that command response.

### Command responses

A **command response** is a value of type `Fw.CmdResponse`.

`Fw.CmdResponse` must be defined in the dictionary, otherwise, an error is raised.
TODO gracefully handle when it's missing.

## Builtin functions
A **builtin function** is a callable whose behavior is explicit in the specification.

### `exit`
#### Signature:

`exit(exit_code: U8)`

#### Semantics
At evaluation:
1. If `exit_code` evaluates to a non-zero value, display that value to the user.
2. The program is halted.

### `log`
#### Signature

`log(operand: F64) -> F64`

#### Semantics

At evaluation:
1. If `operand` is outside the domain of the natural logarithm function, halt the program and display an error code.
2. The expression evaluates to the natural logarithm of `operand`.

### `sleep`
#### Signature

`sleep(seconds: U32, microseconds: U32)`

#### Semantics

At evaluation, the program [sleeps](#sleeping) for a duration of `seconds` seconds and `microseconds` microseconds.

### `sleep_until`
#### Signature

`sleep_until(wakeup_time: Fw.Time)`

#### Semantics

At evaluation, the program [sleeps](#sleeping) until the given `wakeup_time`.

### `now`
#### Signature
`now() -> Fw.Time`
#### Semantics

At evaluation, the function call evaluates to a `Fw.Time` value representing the current time.
TODO this really should be linked to the FpySequencer spec to say exactly where it gets this, etc. we also need engineering details

### `iabs`
#### Signature
`iabs(value: I64) -> I64`

#### Semantics
At evaluation, the function call evaluates to the absolute value of `value`.

TODO specify what happens if the abs value is outside of i64

### `fabs`
#### Signature
`fabs(value: F64) -> F64`

#### Semantics
At evaluation, the function call evaluates to the absolute value of `value`.
TODO specify what happens if the abs value is outside of i64

## Builtin libraries


## Time functions
TODO Fpy provides builtin functions for comparing and manipulating time values:

* `time_cmp(lhs: Fw.Time, rhs: Fw.Time) -> I8`: compares two absolute times. Returns `-1` if `lhs` occurs before `rhs`, `0` if they are the same moment, `1` if `lhs` occurs after `rhs`, or `2` if the time bases differ (incomparable).
* `time_interval_cmp(lhs: Fw.TimeIntervalValue, rhs: Fw.TimeIntervalValue) -> I8`: compares two time intervals. Returns `-1` if `lhs` is a shorter duration than `rhs`, `0` if they are the same duration, or `1` if `lhs` is a longer duration than `rhs`.
* `time_sub(lhs: Fw.Time, rhs: Fw.Time) -> Fw.TimeIntervalValue`: subtracts two absolute times, producing a time interval. Asserts that both times have the same time base and that `lhs` occurs after `rhs` (no negative intervals).
* `time_add(lhs: Fw.Time, rhs: Fw.TimeIntervalValue) -> Fw.Time`: adds a time interval to an absolute time, producing a new absolute time. Asserts that the result does not overflow.

These functions are implemented in Fpy itself (see `src/fpy/builtin/time.fpy`) and are automatically available in all sequences.

## Constructors
TODO Structs, arrays, and `Fw.Time` expose constructors whose callable name is the fully qualified type name. Their arguments correspond to the members in definition order (struct fields by name, array elements as `e0`, `e1`, ..., and `Fw.Time` with `time_base`, `time_context`, `seconds`, `useconds`). A constructor call serializes the provided values into a new instance of that type.

## Numeric casts
TODO Each concrete numeric type provides a callable whose name matches the type (for example `U16(value)` or `F64(value)`). Casts accept exactly one numeric argument. Unlike implicit coercion, casts always force the operand into the target type even when this requires narrowing; range checks are suppressed and the value is truncated or rounded if necessary. See [Casting](#casting) for details.

# Types

A **type** is a set of **values**.

The values of a type are unique to that type.

> In other words, there are no union types and there is no type inheritance.

New types cannot be defined by the program.

A **serializable type** is a type whose values can be expressed in a binary format.

A **constant-sized type** is a serializable type whose binary form always has the same length in bytes.

A **numeric type** is a [primitive numeric type](#primitive-numeric-types), or the [internal Int or Float](#internal-types) types.

> Right now, the only serializable but non-constant-sized type are the [dictionary string](#dictionary-strings) types.

Types can be divided into three categories:
* Primitive types
* Internal types
* Dictionary types

## Primitive types

**Primitive types** are types which are always present in the global scope.

> That is, they do not have to be in the F-Prime dictionary to be referenced by name in the program.
> Because they are present in the global scope, we will use their associated name in the global scope to refer to them throughout this specification. For instance, when we say type `U16`, we are talking about the type in the global scope with name `U16`.

All primitive types are serializable, constant-sized types.

The list of primitive types is:
* All [primitive numeric types](#primitive-numeric-types)
* The [Boolean type](#boolean-type)

### Primitive numeric types
`U8`, `U16`, `U32`, and `U64` are the primitive unsigned integer types with bitwidths 8, 16, 32 and 64, respsectively. They use the standard binary representation of unsigned integers.

`I8`, `I16`, `I32`, and `I64` are the primitive signed integer types with bitwidths 8, 16, 32 and 64, respsectively. They use the standard two's complement representation of signed integers.

`F32`, and `F64` are the primitive IEEE floating-point types with bitwidths 32 and 64, respectively.

> There are other numerical types such as [Int or Float](#internal-types) which are not primitve.

### Boolean type
`bool` is a primitive type whose only values may be the [Boolean literals](todo) `True` and `False`.

TODO make sure that Fw.Time is counted as a dictionary type, but one which is required to be in the dict?

## Type aliases
**Loop var type** is an alias for `I64`.
**Array index type** is an alias for `I64`.

## Internal types

**Internal types** are types which are never present in the global scope.

> That is, they cannot be referenced by name in the program.

No internal types are serializable types.

**Int** is an internal type whose values are integers of arbitrary precision.

**Float** is an internal type whose values are decimals per the Python [decimal](https://docs.python.org/3/library/decimal.html#module-decimal) implementation.

The precision of Float is 30 decimal places.

**String** is an internal type whose values are strings of arbitrary length.

**Range** is an internal type whose values are pairs of an lower and upper bound of loop var type.

**Nothing** is an internal type which has no values.

## Dictionary types
**Dictionary types** are types defined in the F-Prime dictionary.

> Because the semantics of these types is defined in the FPP specification, there is some overlap here. This specification just addresses the semantics of these types as relevant to Fpy.

All dictionary types are serializable types.

Dictionary types can be divided into three categories:
* [Structs](#structs)
* [Arrays](#arrays)
* [Enums](#enums)
* [Strings](#dictionary-strings)

### Structs
A **struct** is a category of dictionary type defined by an ordered list of members.

A **member** is a pair of a name and a serializable type.

A struct may not have two members with the same name.
TODO is this rule necessary? This is enforced upstream by FPP

The binary form of a struct value is the concatenated binary forms of its member values, in order.

> If any of a struct's members are non-constant-sized types, the struct is a non-constant-sized type.

### Arrays

An **array** is a category of dictionary type defined by a non-negative integer length, and an element type.

The **element type** of an array type is the type of its elements.

An **element** is a value of element type at an index in an array.

The binary form of an array value is the concatenated binary form of its elements, in order.

> If the element type is a non-constant-sized type, the array is a non-constant-sized type.

### Enums

An **enum** is a category of dictionary type whose values are a finite set of enum constants.

An **enum constant** is a pair of a name and a value of the enum's representation type.

An **enum representation type** is the [primitive integer type](#primitive-numeric-types) associated with the enum constants' values.

The binary form of an enum constant is the binary form of its integer value.

### Dictionary strings

A **dictionary string** is a category of dictionary type whose values are strings.

Dictionary strings are non-constant-sized types.

## Fields

A **field-based type** is a type defined by its fields.

A **field** of a type is a name-and-type pair 

An **array** is a category of type with 

A **struct** is a category of type 

is an [array element](#arrays) or a [struct member](#structs).

The **field base** of a field is the first non-field parent of a field.

> For instance, the field base of `a.b.c`, if `a` were a variable and `b` and `c` were fields, would be `a`.

## Populating dictionary types

For each type `T` with fully-qualified name `F.Q.N` encountered in the F-Prime dictionary, that type will be present in the global scope with the same fully-qualified name.

## Type conversion

TODO Type conversion is the process of converting an expression from one type to another. It can either be implicit, in which case it is called coercion, or explicit, in which case it is called casting.

### Intermediate types

The **intermediate type** of a binary or unary operator expression is the type to which all argument expressions will be coerced to.

Intermediate types are picked via the following rules:

1. The intermediate type of Boolean operators is always `bool`.
2. The intermediate type of `==` and `!=` may be any type, so long as the left and right hand sides are the same type. If both are numeric then continue.
3. If either argument is non-numeric, raise an error.
4. If the operator is `/` or `**`, the intermediate type is always `F64`.
5. If either argument is a float, the intermediate type is `F64`.
6. If either argument is an unsigned integer, the intermediate type is `U64`.
7. Otherwise, the intermediate type is `I64`.

If the expressions given to the operator are not of the intermediate type, type coercion rules are applied.

## Result type

The result type is the type of the value produced by the operator.
1. For numeric operators, the result type is the intermediate type.
2. For boolean and comparison operators, the result type is `bool`.

Normal type coercion rules apply to the result, of course. Once the operator has produced a value, it may be coerced into some other type depending on context.

# Expressions

An **expression** can be evaluated to produce a value of a type.

A **constant expression** is an expression which can be evaluated without running the program.

## Literals

A **literal** is an expression whose value is explicit in the source code.

All literal expressions are constant expressions.

### Integer literals

#### Decimal literal syntax

Rule:

```
DEC_LITERAL:   "1".."9" ("_"?  "0".."9")*
           |   "0"      ("_"?  "0")* /(?![1-9xX])/
```

#### Hexadecimal literal syntax

Rule:

`HEX_LITERAL: ("0x" | "0X") ("_"? /[0-9a-fA-F]/)+`

#### Semantics

Integer literals have type [Int](#internal-types).

### Float literals

#### Syntax
```
_SPECIAL_DEC: "0".."9" ("_"?  "0".."9")*

DECIMAL: "." _SPECIAL_DEC | _SPECIAL_DEC "." _SPECIAL_DEC
_EXP: ("e"|"E") ["+" | "-"] _SPECIAL_DEC
FLOAT_LITERAL: _SPECIAL_DEC _EXP | DECIMAL _EXP?
```

Float literals have type [Float](#internal-types).

A float literal is rounded to the nearest value of type Float.

### String literals
#### Syntax

Rule:

`STRING_LITERAL: /("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i`

#### Semantics

String literals have type [String](#internal-types).

### Boolean literals
#### Syntax
`BOOLEAN_LITERAL: "True" | "False"`
#### Semantics

Boolean literals have type [`bool`](#boolean-type)

## Member access expression
### Syntax

Rule:

`member_access_expr: expr "." name`

Name:

`member_access_expr: parent "." member`

### Semantics

If `parent` is not an expression, an error is raised.

> Namespaces, types names, and function names are valid expressions syntactically, but not semantically. Thus, trying to access a member of either of these symbols will raise an error.

If the type of `parent` is not a [struct](#structs), an error is raised.

If the type of `parent` is not [constant-sized](#types), an error is raised.

If `member` is not a member of the type of `parent`, an error is raised.

The type of a member access is the type of the `member` in the type of `parent`.

At evaluation:
1. The `parent` is evaluated.
2. The member access expression evaluates to the value of the `member` in the `parent` value.

## Function call expression
### Syntax

Rule:

```
func_call: expr "(" [arguments] ")"`
arguments: argument ("," argument)*
argument: NAME "=" expr -> named_argument
        | expr -> positional_argument
```

Name:

`func_call: func "(" arguments ")"`

`func` is resolved in the callable name group.

All argument expressions are resolved in the value name group.

### Semantics

At evaluation:
1. Each argument is evaluated from left to right.
2. The behavior defined in the semantics for the [callable](#callables) referenced by `func` is 

## Binary operator expressions

A **binary operator expression** is an expression with a left and right-hand expression, and a binary operator in between, which acts on both values to produce a new value.

The list of **binary operators** is:
* The [addition operator](#subtraction-semantics) `+`
* The [subtraction operator](#multiplication-semantics) `-`
* The [multiplication operator](#multiplication-semantics) `*`
* The [division operator](#division-semantics) `/`
* The [floor division operator](#floor-division-semantics) `//`
* The [modulus operator](#modulus-semantics) `%`
* The [exponentiation operator](#exponentiation-semantics) `**`
* The [Boolean operators](#boolean-operator-semantics) `and` and `or`
* The [comparison operators](#comparison-semantics) `>`, `>=`, `<`, and `<=`
* The [equality operator](#equality-semantics) `==`
* The [inequality operator](#inequality-semantics) `!=`
* The [range operator](#range-semantics) `..`

### Syntax

Rule:

`binary_op: expr BINARY_OP expr`

Name:

`binary_op: lhs op rhs`

`lhs` and `rhs` are resolved in the value name group.

### Semantics

For each use of a binary operator, an [intermediate type](#intermediate-types) is picked, as described in the operator's semantics.

If `lhs` or `rhs` cannot be [coerced](#type-coercion) into the intermediate type, an error is raised.

If `lhs` and `rhs` are constant expressions, the binary operator expression is a constant expression.

At evaluation, for all operators besides the [Boolean operators](#boolean-operator-semantics):
1. `lhs` is evaluated and coerced into the intermediate type.
2. `rhs` is evaluated and coerced into the intermediate type.
3. The expression evaluates to a value of the intermediate type, as described in the operator's semantics.

#### Addition semantics
The addition operator is `+`.

If neither `lhs` nor `rhs` are expressions of a [numeric type](#types), an error is raised.

The expression evaluates to the result of adding 

#### Subtraction semantics
#### Multiplication semantics

These operators require numeric operands and produce a result in the chosen intermediate type. Addition, subtraction, and multiplication differ only in which arithmetic operation they perform. Integer overflow wraps according to the destination type when the result is ultimately stored, and floating-point operations follow IEEE-754 behavior.

#### Division semantics
Both operands are promoted to `F64`, and the result is always an `F64`. This means you must explicitly cast the result to store it in an integer type.

#### Floor division semantics
With integer operands, `//` performs truncating division using the signed or unsigned divide directive. If either operand is a float, the compiler divides in `F64`, converts the quotient to a signed 64-bit integer (which truncates toward zero), and converts back to `F64`, so floating-point floor division also truncates toward zero.

### Modulus semantics
Modulus works for numeric operands. Signed operands use the signed modulo directive, unsigned operands use the unsigned directive, and floats use floating-point modulo. For signed integers the remainder has the same sign as the dividend.

#### Exponentiation semantics
Both operands are coerced to `F64`, the exponentiation happens in floating point, and the result type is `F64`.

#### Boolean operator semantics
Operands must be `bool`. `not` negates a single operand. `and` evaluates the left operand first and only evaluates the right operand when the left operand is `True`. Conversely, `or` skips the right operand when the left operand is `True`. The result of every boolean operator is `bool`.

#### Comparison semantics
Inequalities require numeric operands. Each operand is coerced to the intermediate type, the comparison runs in that type, and the result is `bool`.

#### Equality semantics
If both operands are numeric, equality uses the same intermediate-type rules as arithmetic operators. Otherwise both operands must have the exact same concrete type (struct, array, enum, or `Fw.Time`). The compiler compares their serialized bytes. Strings cannot be compared.

#### Range semantics
The range operator is `..`.

If `lhs` or `rhs` cannot be coerced to [loop var type](#type-aliases), an error is raised.

#### Order of operations
The order in which operations take precedence, from most strongly binding to least strongly binding, is:
1. [Exponentiation](#exponentiation-semantics)
2. [Negation](#negation-operator-semantics) and [identity](#identity-operator-semantics)
3. [Multiplication](#multiplication-semantics), [division](#division-semantics), [floor division](#floor-division-semantics), and [modulus](#modulus-semantics)
4. [Addition](#addition-semantics) and [subtraction](#subtraction-semantics)
5. [Range](#range-semantics)
6. [Comparison](#comparison-semantics)
7. [Not](#boolean-operator-semantics)
8. [And](#boolean-operator-semantics)
9. [Or](#boolean-operator-semantics)

If two operators have the same precedence in the above list, then the leftmost operator binds more strongly.

## Unary operators
### Syntax

Rule:

`unary_op: expr OP`

Name:

`unary_op: val op`

### Negation operator semantics
### Identity operator semantics

## Intermediate types

The **intermediate type** of an operator expression is the type to which the operator's sub-expressions are [coerced](#type-coercion) to.

If any sub-expression

### Numeric intermediate types

The numeric type hierarchy is as follows:
* 



        # we split this algo up into two stages: picking the type category (float, uint or int), and picking the type bitwidth

        # pick the type category:
        type_category = None
        if op == BinaryStackOp.DIVIDE or op == BinaryStackOp.EXPONENT:
            # always do true division and exponentiation over floats, python style
            # this is because, for the given op, even with integer inputs, we might get
            # float outputs
            type_category = "float"
        elif any(issubclass(t, FloatValue) for t in arg_types):
            # otherwise if any args are floats, use float
            type_category = "float"
        elif any(t in UNSIGNED_INTEGER_TYPES for t in arg_types):
            # otherwise if any args are unsigned, use unsigned
            type_category = "uint"
        else:
            # otherwise use signed int
            type_category = "int"

        # pick the bitwidth
        # we only use the arb precision types for constants, so if theyre all arb precision, they're consts
        constants = all(t in ARBITRARY_PRECISION_TYPES for t in arg_types)

        if constants:
            # we can constant fold this, so use infinite bitwidth
            if type_category == "float":
                return FpyFloatValue
            assert type_category == "int" or type_category == "uint"
            return FpyIntegerValue

        # can't const fold
        if type_category == "float":
            return F64Value
        if type_category == "uint":
            return U64Value
        assert type_category == "int"
        return I64Value


## Type conversion

**Type conversion** is the process by which values of one type are converted into values of another type.

There are two kinds of type conversion:
* [Casting](#casting)
* [Coercion](#type-coercion)

Type casting is merely an explicit flag for type coercion to take place


### Type coercion
**Type coercion** is type conversion that happens implicitly to an expression when required by that expression's semantic context.


Coercion happens when an expression of type *A* is used in a syntactic element which requires an expression of type *B*. For example, functions, operators and variable assignments all require specific input types, so type coercion happens in each of these.
In general, the rule of thumb is that coercion is allowed if the destination type can represent all possible values of the source type, with some exceptions. The following rules determine when type coercion can be performed:

1. If the source and destination types are identical, no coercion is performed.
2. *LiteralString* values may be coerced into any FPP string type. No other string expression can be coerced.
3. Otherwise both source and destination must be numeric (`NumericalValue`). Numeric coercions obey these constraints:
    * Floats never coerce to integers.
    * Integers may always coerce to floats.
    * Float-to-float coercions require a destination bit width greater than or equal to the source width.
    * Integer-to-integer coercions require matching signedness and a destination bit width greater than or equal to the source width.
    * Arbitrary-precision types (`Int`/`Float`) may coerce to any finite-width numeric type.
If no rule matches, the compiler raises an error.

Compile-time constant floats (including literals and constant-folded expressions) can only be narrowed into a smaller floating-point type when the value lies inside the destination’s representable range. When the value fits, the compiler rounds it to the nearest representable floating-point number; otherwise compilation fails with an out-of-range error.


# Execution

## Control flow
A **branch** is a block of code which conditionally executes.

> That is, whether or not that code exeutes depends on some expression.

The list of statements which have branches is:
* The [if statement](#ifs)
* The [while loop statement](#while-loop-statement)
* The [for loop statement](#for-loop-statement)
* The [check statement](#check-statement)
TODO how does this defn handle assert/exit?

## Sleeping
The program may **sleep** until an absolute time called the **wakeup time**.

While the current time is before the wakeup time, no statements may execute

The program may also sleep for a time duration. This is the same as sleeping with a wakeup time of now, plus the specified duration.
