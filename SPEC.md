> **WARNING:** The Fpy specification is a work-in-progress

# Fpy Specification

Fpy is a sequencing language for the F-Prime flight software framework. It combines Python-like syntax with the FPP model, with several domain-specific features for spacecraft operation.

This document specifies the syntax and semantics of the Fpy sequencing language. In other words, it specifies which programs the compiler should accept, and how the output should behave when run. The intent is to leave the implementation of the compiler, the bytecode it generates, and the virtual machine it runs on, unspecified. 

It is assumed the reader is familiar with the FPP model, including commands, telemetry, parameters, structs, arrays and enums.

> Informal notes and explanations are quoted like this

Terms are **bolded** in the location they are primarily defined.

`Monospaced` terms refer to syntactic rules, type names or example Fpy code.

Monospaced code blocks are example Fpy code:
```py
def example(fpy: U8):
    return
```

# Syntax
The syntax of Fpy is defined using the [Lark grammar syntax](https://lark-parser.readthedocs.io/en/stable/grammar.html), in [grammar.lark](grammar.lark).

The rest of the specification is dedicated to the semantics of Fpy.

# Symbols

A **symbol** is a language construct that can be referred to by some name in the program. 

The following language constructs may be symbols:
* [namespaces](#namespaces)
* [variables](#variables)
* [functions](#functions)
* [types](#types)
* telemetry channels
* parameters
* enum constants

# Scopes

A **scope** is a mapping of names to symbols, accessed via some region of the source code.

The **global scope** is the scope accessible throughout the entire source code.

Each [function](#functions) has a **function scope**, accessible in its body.

The **resolving scope** is the most specific scope that some part of the source code has access to.

# Name resolution
If a name is prefixed with a dot, perform attribute resolution
Otherwise, resolve the name in the resolving scope and resolving name group.

If resolution fails, and the resolving scope is not the global scope,


# Name groups

Scopes are divided into **name groups**.

The list of name groups is:
* The **[value](#values) name group**
* The **[type](#types) name group**
* The **[callable](#callables) name group**

Each name group only contains names which map to their particular language construct, or namespaces with the same property, recursively.

Name groups do not intersect.

> This means that the names of callables, types and values never conflict. 

Name groups are accessed via context.

The **resolving name group** is the name group that a name should be resolved in, based on its context.


# Namespaces

A **namespace** is a mapping of names to symbols, associated with a name.

To access a namespace, a special case of `get_attr` is used:
* `parent` must resolve to 



A **qualified name** is one of:
* A name
* A qualified name, followed by a `.`, followed by a name

## Name resolution

Given a resolving scope and resolving name group, a

# definitions

A **definition** is a language construct that introduces a name-to-symbol mapping to a scope and name group.

The list of definitions is:
* [Variable definitions](#variable-definition)
* [Function definitions](#function-definition)

# Variables

A **variable** is a [symbol](#symbols) associated with a static [type](#types) and a mutable [value](#values).

## Variable definition

A **variable definition** introduces a name-to-[variable](#variables) mapping to its resolving scope.

### Syntax
Rule:

`variable_declare_stmt: name ":" expr "=" expr`

Name:

`lhs ":" type_ann "=" rhs`

`lhs` and `rhs` are resolved in the value name group.

`type_ann` is resolved in the type name group.

### Semantics

If `lhs` resolves to a previously-defined variable, an error is raised.

If `rhs` cannot be coerced to type `type_ann`, an error is raised.

The new variable has name `lhs` and type `type_ann`. It is added to the resolving scope.

At execution, `rhs` is [evaluated](#evaluation) and [coerced](#type-coercion) to type `type_ann`. This becomes the variable's initial value.

After execution, the variable is considered defined.

## Variable assignment

A **variable assignment** mutates the value of a variable.

### Syntax
Rule:

`variable_assign_stmt: name "=" expr`

Name:

`lhs "=" rhs`

`lhs` and `rhs` are resolved in the value name group.

### Semantics

If `lhs` is not a variable, an error is raised.

If the variable has not been defined yet, an error is raised.

If `rhs` cannot be coerced to the variable's type, an error is raised.

At execution, `rhs` is [evaluated](#evaluation) and [coerced](#type-coercion) to the variable's type. This becomes the variable's new value.

## Member assignment

A **member assignment** mutates the value of a [member](todo) in a variable.

### Syntax
Rule:

`variable_assign_member_stmt: expr "." name "=" expr`

Name:

`parent "." member "=" rhs`

`parent`, and `rhs` are resolved in the value name group.

### Semantics

If `parent` is not a variable or a [field](#fields) with a [field base](todo) that is a variable, an error is raised.

> This allows for setting a field of a field to arbitrary depth, as long as the underlying thing you're modifying is a variable.

If `parent`'s type is:
* not [constant-sized](todo), or
* not a [struct](#structs) type, or
* `member` is not a member of `parent`'s type

... an error is raised.

If the variable has not been defined yet, an error is raised.

If `rhs` cannot be coerced to the member's type, an error is raised.

At execution, `rhs` is [evaluated](#evaluation) and [coerced](#type-coercion) to the member's type. This becomes the member's new value.

The value of the variable is unchanged except for the member.

## Element assignment

An **element assignment** mutates the value of an [element](todo) in a variable.

### Syntax
Rule:

`variable_assign_element_stmt: expr "[" expr "]" "=" expr`

Name:

`parent "[" item "]" "=" rhs`

`parent`, `item` and `rhs` are resolved in the value name group.

### Semantics

If `parent` is not a variable or a [field](#fields) with a [field base](todo) that is a variable, an error is raised.

> This allows for setting a field of a field to arbitrary depth, as long as the underlying thing you're modifying is a variable.

If the variable has not been defined yet, an error is raised.

If `item` cannot be [coerced](#type-coercion) to [array index type](#type-aliases), an error is raised.

If `parent`'s type is:
* not [constant-sized](todo), or
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

The value produced by [evaluating](todo) a variable is the value most recently assigned to that variable, or the initial value if it has only been defined.

If a variable is evaluated before it has been defined, an error is raised.

# Functions

A **function** is a [callable](todo) [symbol](#symbols) with an inner scope, parameters, code and a return [type](#types).

## Parameters

A **parameter** is a [variable](#variables) implicitly defined by a function in that function's scope.

When a function is [called](todo), each parameter is set to an initial value.

The initial value may either be from an [argument](todo), or a default value, if one is specified in the function definition.

> In all other respects, parameters are like normal variables.

## Function definition

A **function definition** introduces a name-to-[function](#function) mapping to the global scope.

### Syntax
Rule:

`function_def_stmt: "def" name "(" [parameters] ")" ["->" expr] ":" block`

`parameters: parameter ("," parameter)*`

`parameter: name ":" expr ["=" expr]`

Name:

`"def" name "(" parameters ")" "->" return_type ":" body`

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

The new function with name `name` is added to the global scope.

> Because functions can only be defined in the global scope, you cannot declare a function in a function.

> Functions can be used before they are defined.

## Function evaluation
Functions can be evaluated by [calling](todo) the function.

When a function is called:
1. Argument values are assigned to parameters
2. The function body executes

During the execution of the function body, if a [return](todo) statement is reached, the function evaluates to the return value of that statement, or no value if the return did not have one.

# Types

A *type* is a set of *values*.

The values of a type are unique to that type.

> In other words, there are no union types and there is no type inheritance.

New types cannot be defined by the program.

A *serializable type* is a type whose values can be expressed in a binary format.

Types can be divided into three categories:
* Primitive types
* Internal types
* Dictionary types

## Primitive types

*Primitive types* are types which are always present in the global type scope.

> That is, they do not have to be in the F-Prime dictionary to be referenced by name in the program.
> Because they are present in the global type scope, we will use their associated name in the global type scope to refer to them throughout this specification. For instance, when we say type `U16`, we are talking about the type in the global type scope with name `U16`.

All primitive types are serializable types.

The list of primitive types is:
* All [primitive numeric types](#primitive-numeric-types)
* The [Boolean type](#boolean-type)

### Primitive numeric types
`U8`, `U16`, `U32`, and `U64` are the primitive unsigned integer types, with bitwidths 8, 16, 32 and 64, respsectively. They use the standard binary representation of unsigned integers.

`I8`, `I16`, `I32`, and `I64` are the primitive signed integer types, with bitwidths 8, 16, 32 and 64, respsectively. They use the standard two's complement representation of signed integers.

`F32`, and `F64` are the primitive IEEE floating-point types, with bitwidths 32 and 64, respectively.

### Boolean type
`bool` is the Boolean type, whose only values may be the [Boolean literals](todo) `True` and `False`.

TODO make sure that Fw.Time is counted as a dictionary type, but one which is required to be in the dict?

## Type aliases
*Loop var type* is an alias for `I64`.
*Array index type* is an alias for `I64`.

## Internal types

*Internal types* are types which are never present in the global type scope.

> That is, they cannot be referenced by name in the program.

No internal types are serializable types.

*Int* is an internal type whose values are integers of arbitrary precision.

*Float* is an internal type whose values are decimals per the Python [decimal](https://docs.python.org/3/library/decimal.html#module-decimal) implementation.

The precision of Float is 30 decimal places.

*String* is an internal type whose values are strings of arbitrary length.

*Range* is an internal type whose values are pairs of values of loop var type.

*Nothing* is an internal type which has no values.

## Dictionary types
*Dictionary types* are types defined in the F-Prime dictionary.

> Because the semantics of these types is defined in the FPP specification, there is some overlap here. This specification just addresses the semantics of these types as relevant to Fpy.

All dictionary types are serializable types.

Dictionary types can be divided into three categories:
* Structs
* Arrays
* Enums

### Structs
A *struct* is a dictionary type defined by an ordered list of *members*.

A *member* is a pair of a name and a serializable type.

A struct may not have two members with the same name.
TODO is this rule necessary? This is enforced upstream by FPP

The binary form of a struct value is the concatenated binary forms of its member values, in order.

### Arrays

An *array* is a dictionary type defined by a non-negative integer length, and an *element type*.

The *element type* of an array type is the type of its *elements*.

An *element* is an value associated with an index in an array.

The binary form of an array value is the concatenated binary form of its element values, in order.

### Enums

An *enum* is a dictionary type whose values are a finite set of *enum constants*.

An *enum constant* is a pair of a name and a serializable integer value.

The binary form of an enum constant is the binary form of its associated integer value.

## Populating dictionary types

For each type `T` with fully qualified name `A.B.C` encountered in the F-Prime dictionary:
1. Map name `C` to `T` in namespace `B`.
2. Map name `B` to namespace `B` in namespace `A`.
3. Map name `A` to namespace `A` in the global type scope.

Each qualifier becomes a namespace, and the final name maps to the type.

Basically: map the type to the last component of the fully qualified name. Then construct namespaces for each of the other components

# Name resolution

*Name resolution* is the process of

## Type name resolution
To resolve a `type_name`:
1. Start in the global type scope.
2. For each name from left to right, look up 

# Qualified name
A *qualified name* is one of the following:
1. A name
2. `Q.N`, where `Q` is a qualified name and `N` is a name.

## Qualified name resolution
A qualified name is a series of names corresponding to namespaces separated by dots, followed by a name of a symbol in the final namespace.
For a given qualifier `Q` and name `N` and scope `S`, the fully-qualified name is resolved to a symbol as follows:
1. Resolve the qualifier
1. Look up

# Attributes

An *attribute access* is a use of the the `get_attr` syntactic rule.

The *parent* of an attribute access is the expression to the left of the dot.

The *attribute* of an attribute access is the string to the right of the dot.

# Qualified name

A *qualified name* is a name of 


To resolve a name to a [value](#values):
1. If the name is inside a function, check the function's value scope.
2. Check the global value scope.

At the end, if the name is not found, an error is raised.

To resolve a name to a [callable](#callables), the global callable scope is checked, and an error is raised if the name is not found.
TODO: but this isn't really true right because we can resolve more than just names to callables? maybe this section isn't needed?


To resolve a name to a [type](#callables), the global callable scope is checked, and an error is raised if the name is not found.



## Attribute resolution

An attribute may resolve to a 

To resolve an attribute to a symbol, first the parent is resolved, recursively.

The parent of an attribute may be one of the following:
* A namespace
* A type
* An [expression](todo)

An attribute access has one of many different behaviors depending on the parent:

| Parent category | Attribute behavior |
|---|---|
|Expression|Member access|
|Namespace|Name lookup|
|Type|Raise an error|
|Callable|Raise an error|


## Fields
Fields refer to either a member of a struct, or an element of an array. Field access uses Python-like syntax: `expr.member` reads a struct (or `Fw.Time`) member and `expr[index]` reads an array element. These operations are only legal when the referenced type has a statically known layout. Because strings do not have a fixed size in memory, structs or arrays with string fields do not have a statically known layout.

Accessing `Fw.Time` produces synthetic members named `time_base`, `time_context`, `seconds`, and `useconds` with the types defined by F´.

Array indices are coerced to `I64` before use. If the index is a compile-time constant the compiler emits an error when it falls outside `[0, length)`. Otherwise the generated bytecode performs a runtime bounds check and terminates the sequence with `DirectiveErrorCode.ARRAY_OUT_OF_BOUNDS` if it fails.

# Functions

Every callable in Fpy uses the same syntax:
```
function_name(arg_0, arg_1, ..., arg_n)
```
Arguments are evaluated left-to-right exactly once. After evaluation, the compiler coerces each argument to the parameter type declared by the callable (except when invoking an explicit numeric cast, which bypasses the usual coercion rules). If any coercion fails, compilation fails. The value produced by the call has the callable’s declared return type and may later be coerced again by the surrounding context.

Fpy exposes several categories of callables:

## Commands
Every command instance defined in the FPP dictionary can be called. The callable name is the command’s fully qualified name, the signature matches the command’s FPP arguments, and the return type is always `Fw.CmdResponse`. Calling a command immediately serializes the opcode and arguments, sends them to the dispatcher, blocks the sequence until the command finishes, and then yields the dispatcher’s `Fw.CmdResponse`.

## Macros
Inline macros behave like functions whose bodies are pre-defined sequences of bytecode directives. They are defined in `src/fpy/macros.py`, evaluate their arguments, push those values onto the stack, and then emit the directives listed below.

Available macros:

* `exit(exit_code: U8)`: terminates the sequence immediately by emitting an `ExitDirective`.
* `log(operand: F64) -> F64`: computes the natural logarithm of the operand using `FloatLogDirective` and leaves the `F64` result on the stack.
* `sleep(seconds: U32, microseconds: U32)`: waits for the specified relative duration (the assembler emits `WaitRelDirective`).
* `sleep_until(wakeup_time: Fw.Time)`: waits until the supplied absolute time using `WaitAbsDirective`.
* `now() -> Fw.Time`: pushes the current time via `PushTimeDirective`.
* `iabs(value: I64) -> I64`: returns the absolute value of a signed 64-bit integer.
* `fabs(value: F64) -> F64`: returns the absolute value of a 64-bit float.

## Time functions
Fpy provides builtin functions for comparing and manipulating time values:

* `time_cmp(lhs: Fw.Time, rhs: Fw.Time) -> I8`: compares two absolute times. Returns `-1` if `lhs` occurs before `rhs`, `0` if they are the same moment, `1` if `lhs` occurs after `rhs`, or `2` if the time bases differ (incomparable).
* `time_interval_cmp(lhs: Fw.TimeIntervalValue, rhs: Fw.TimeIntervalValue) -> I8`: compares two time intervals. Returns `-1` if `lhs` is a shorter duration than `rhs`, `0` if they are the same duration, or `1` if `lhs` is a longer duration than `rhs`.
* `time_sub(lhs: Fw.Time, rhs: Fw.Time) -> Fw.TimeIntervalValue`: subtracts two absolute times, producing a time interval. Asserts that both times have the same time base and that `lhs` occurs after `rhs` (no negative intervals).
* `time_add(lhs: Fw.Time, rhs: Fw.TimeIntervalValue) -> Fw.Time`: adds a time interval to an absolute time, producing a new absolute time. Asserts that the result does not overflow.

These functions are implemented in Fpy itself (see `src/fpy/builtin/time.fpy`) and are automatically available in all sequences.

## Type constructors
Structs, arrays, and `Fw.Time` expose constructors whose callable name is the fully qualified type name. Their arguments correspond to the members in definition order (struct fields by name, array elements as `e0`, `e1`, ..., and `Fw.Time` with `time_base`, `time_context`, `seconds`, `useconds`). A constructor call serializes the provided values into a new instance of that type.

## Numeric casts
Each concrete numeric type provides a callable whose name matches the type (for example `U16(value)` or `F64(value)`). Casts accept exactly one numeric argument. Unlike implicit coercion, casts always force the operand into the target type even when this requires narrowing; range checks are suppressed and the value is truncated or rounded if necessary. See [Casting](#casting) for details.

## User-defined functions
A function definition introduces a new callable into scope. Function definitions must appear at the top level of the sequence; a function definition nested inside another function, a loop body, or a conditional branch is a compile-time error. The syntax is:
```
def name(param_0: Type0, param_1: Type1 = default_value, ...) [-> ReturnType]:
    body
```

### Parameters
Each parameter definition consists of a name followed by a colon and a type annotation. A parameter may optionally include a default value, written as `= expr` after the type annotation. Default value expressions must be constant expressions: literals, enum constants, or type constructors whose arguments are themselves constant expressions. Expressions referencing telemetry channels, variables, or function calls are not constant and produce a compile-time error when used as defaults.

Arguments may be passed by position or by name. Positional arguments are bound to parameters left-to-right. Named arguments use the syntax `name=expr` and bind the value of `expr` to the parameter with the matching name. All positional arguments must precede all named arguments. A parameter may not be bound more than once; supplying both a positional argument and a named argument for the same parameter is a compile-time error. If fewer arguments are supplied than parameters, the remaining parameters must have default values; those defaults are evaluated and bound. Supplying more positional arguments than parameters, or naming a parameter that does not exist, is a compile-time error.

### Return type
The return type annotation `-> Type` is optional. When present, every control-flow path through the function body must terminate with a `return expr` statement where `expr` has a type coercible to `Type`. When absent, the function does not produce a value; `return` statements in such functions must not include an expression, and the call expression has no usable result.

### Scope
A function body introduces a new scope. Within this scope the following names are visible:
1. Parameters declared in the function signature.
2. Local variables declared within the function body.
3. Top-level variables declared before the call site of the function (not the definition site).
4. All dictionary objects: commands, telemetry channels, parameters, enum constants, and types.
5. All user-defined functions, including functions defined after the current function (forward references are permitted).

Assignments to top-level variables within a function body modify the original variable. Assignments to parameters or local variables do not affect any outer scope.

# Type conversion

Type conversion is the process of converting an expression from one type to another. It can either be implicit, in which case it is called coercion, or explicit, in which case it is called casting.

## Type coercion
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

## Casting
Each finite-bitwidth numeric type exposes an explicit cast with the same name as the type, e.g. `U32(value)` or `F64(value)`. Casts accept any numeric expression and bypass the implicit-coercion restrictions above: the operand is forced to the target type even when that entails narrowing, and compile-time range checks are suppressed. No casts exist for structs, arrays, enums, strings, or `Fw.Time`.

# Expressions

An *expression* is a syntactic construct which can be *evaluated*.

*Evaluation* is the process of converting an expression to a value.



## Integer literals

Integer literals have type *Integer*, which is not directly referenceable by the user. The *Integer* type supports integers of arbitrary size.

## Float literals

Float literals have type *Float*, which is not directly referenceable by the user. The *Float* type supports up to 30 decimal points of precision. It is implemented with the Python `Decimal` type.

## String literals
String literals are strings matching:
```
STRING: /("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i
```

They have type *LiteralString*, which is not directly referenceable by the user. The *LiteralString* type supports strings of arbitrary length.


# Operators

Fpy supports the following operators:
* Basic arithmetic: `+, -, *, /`
* Modulo: `%`
* Exponentiation: `**`
* Floor division: `//`
* Boolean: `and, or, not`
* Comparison: `<, >, <=, >=, ==, !=`

Each time an operator is used, an intermediate type must be picked and both args must be converted to that type.

## Behavior of operators
All operators share the following rules:

1. The left operand is evaluated first, then the right operand. Boolean `and`/`or` short-circuit, so the right operand is skipped when the result is already known.
2. Each operand is coerced to the operator’s intermediate type (see [Intermediate Types](#intermediate-types)). If no valid intermediate type exists, compilation fails.

The subsections below describe behaviors that differ from the general rules.

### Numeric arithmetic (`+`, `-`, `*`)
These operators require numeric operands and produce a result in the chosen intermediate type. Addition, subtraction, and multiplication differ only in which arithmetic operation they perform. Integer overflow wraps according to the destination type when the result is ultimately stored, and floating-point operations follow IEEE-754 behavior.

### True division (`/`)
Both operands are promoted to `F64`, and the result is always an `F64`. This means you must explicitly cast the result to store it in an integer type.

### Floor division (`//`)
With integer operands, `//` performs truncating division using the signed or unsigned divide directive. If either operand is a float, the compiler divides in `F64`, converts the quotient to a signed 64-bit integer (which truncates toward zero), and converts back to `F64`, so floating-point floor division also truncates toward zero.

### Modulo (`%`)
Modulo works for numeric operands. Signed operands use the signed modulo directive, unsigned operands use the unsigned directive, and floats use floating-point modulo. For signed integers the remainder has the same sign as the dividend.

### Exponentiation (`**`)
Both operands are coerced to `F64`, the exponentiation happens in floating point, and the result type is `F64`.

### Boolean operators (`and`, `or`, `not`)
Operands must be `bool`. `not` negates a single operand. `and` evaluates the left operand first and only evaluates the right operand when the left operand is `True`. Conversely, `or` skips the right operand when the left operand is `True`. The result of every boolean operator is `bool`.

### Inequalities (`<`, `<=`, `>`, `>=`)
Inequalities require numeric operands. Each operand is coerced to the intermediate type, the comparison runs in that type, and the result is `bool`.

### Equality (`==`, `!=`)
If both operands are numeric, equality uses the same intermediate-type rules as arithmetic operators. Otherwise both operands must have the exact same concrete type (struct, array, enum, or `Fw.Time`). The compiler compares their serialized bytes. Strings cannot be compared.

## Intermediate types

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

# Loops

## Range expressions

The `lower .. upper` operator produces a `RangeValue`. Both bounds are coerced to `I64`. Range expressions are only meaningful as the right-hand side of a `for` loop, and both bounds are evaluated exactly once.

## For loops

```
for <var> in <lower> .. <upper>:
    <body>
```

* `<lower>` and `<upper>` are evaluated before the loop starts and stored in hidden variables.
* The loop variable is assigned `<lower>`, coerced to `I64`. If the name was not previously declared, the compiler declares it with type `I64`; otherwise the existing variable must already be of that type.
* The body runs while the loop variable is strictly less than `<upper>`. After each iteration the compiler inserts `var = var + 1`. Modifying `var` manually inside the body affects the next iteration in addition to this implicit increment.
* `break` and `continue` statements are only legal inside the loop body and behave as in C: `break` exits the loop, `continue` skips to the implicit increment/check sequence.

## While loops

```
while <condition>:
    <body>
```

The condition is coerced to `bool` and re-evaluated before every iteration. `break` and `continue` are legal only within the loop body.

# Check statement

```
check <condition_expr> [timeout <timeout_expr>] [persist <persist_expr>] [every <every_expr>]:
     <check_body>
[timeout:
     <timeout_body>]
```

While not timed out, evaluate the condition at a given frequency, running the main body if the condition persists true for a given amount of time. Upon timeout, run the timeout body (if provided).

`condition_expr` must be type `bool`
`timeout_expr` (optional) may be type `Fw.Time`, in which case the timeout happens at an absolute time, or `Fw.TimeIntervalValue`, in which case the timeout happens relative to the moment the expression is evaluated. If omitted, the check runs indefinitely until the condition persists for the required duration.
`persist_expr` (optional) must be type `Fw.TimeIntervalValue`. If omitted, defaults to a 0 second interval, meaning the condition only needs to be true once.
`every_expr` (optional) must be type `Fw.TimeIntervalValue`. If omitted, defaults to a 1 second interval.

The `timeout:` clause and its body are optional. If omitted, no action is taken when the check times out.

