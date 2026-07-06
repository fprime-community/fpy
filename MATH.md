# Mathematical specification of Fpy

## Scope

This document defines the semantics of all mathematical actions and builtin functions in Fpy, including:
* Arithmetic
* Truncation, extension
* Casting
* Overflow, underflow
* NaN propagation
* Mathematical library functions


## Numerical types

`U8`, `U16`, `U32`, and `U64` are the primitive unsigned integer types with bitwidths 8, 16, 32 and 64, respectively. They use the standard binary representation of unsigned integers.

`I8`, `I16`, `I32`, and `I64` are the primitive signed integer types with bitwidths 8, 16, 32 and 64, respectively. They use the standard two's complement representation of signed integers.

`F32`, and `F64` are the primitive IEEE floating-point types with bitwidths 32 and 64, respectively.

## Casts

A cast is an expression which converts a value `V` of numerical type `S` to a value `R` of numerical type `T`.

Casts are guaranteed not to end the program, and have no undefined behavior.



If T is a float type, then R is the nearest representable

WIP
Integer to float and float to float rounds to nearest float (overflow just rounds to highest representable float?)
Float to integer overflow saturates the integer
Integer to integer overflow wraps
The reason arithmetic overflow ends the program but cast overflow does not is because casts are explicit, so the programmer is more aware that they are doing an operation which may involve truncation. If they want to handle an overflow differently than casting does, they can just check for overflow before the cast.

An intermediate type is picked which is 64 bits. Both operands are coerced to the intermediate type. The result type of the operation, before coercion or casting, is also the intermediate type.

Arithmetic on F64
These operations on F64s are fully defined in all cases by IEEE 754 and produce the expected result. These operations never end the program at runtime.

Arithmetic on I64, U64
If the operation is division and the denominator is zero, the program ends with an error.

See https://github.com/rust-lang/rfcs/blob/master/text/0560-integer-overflow.md

Otherwise, let R be the mathematical result of the operation given the two operands.

If R is not representable in the result type, the program ends with an error. This is called an underflow or overflow.


### `ln`
#### Signature

`ln(operand: F64) -> F64`

#### Semantics

At evaluation:
1. If `operand` is outside the domain of the natural logarithm function, halt the program and display an error code.
2. The expression evaluates to the natural logarithm of `operand`.

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



## Type conversion

TODO Type conversion is the process of converting an expression from one type to another. It can either be implicit, in which case it is called coercion, or explicit, in which case it is called casting.

### Intermediate types

The **intermediate type** of a binary or unary operator expression is the type to which all argument expressions will be coerced to.

Intermediate types are picked via the following rules:

1. The intermediate type of Boolean operators is always `bool`.
2. The intermediate type of `==` and `!=` may be any type, so long as the left and right hand sides are the same type. If both are numeric then continue.
3. If either argument is non-numeric, raise an error.
4. If the operator is unary `-` and the argument is an unsigned integer, raise an error: negation is undefined for unsigned types (the result would be negative for every nonzero operand).
5. If the operator is `/` or `**`, the intermediate type is always `F64`.
6. If either argument is a float, the intermediate type is `F64`.
7. If either argument is an unsigned integer, the intermediate type is `U64`.
8. Otherwise, the intermediate type is `I64`.

If the expressions given to the operator are not of the intermediate type, type coercion rules are applied.

## Result type

The result type is the type of the value produced by the operator.
1. For numeric operators, the result type is the intermediate type.
2. For boolean and comparison operators, the result type is `bool`.

Normal type coercion rules apply to the result, of course. Once the operator has produced a value, it may be coerced into some other type depending on context.


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

