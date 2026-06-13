# CreateScopes

A scope is an ordered list of statements at a single, continuous indentation level.

If a scope is the global scope, it has no parent scope.
Otherwise, the parent scope of scope S is the scope at one indentation level lower than S.

If the scope is part of a function definition, it is a function scope.

The enclosing scope of E is the nearest parent scope of E.

# CheckSequenceMetadataDefinedAtTop

For each sequence metadata statement in the root scope:
1. If it is not the first statement in the scope, raise an error.
2. If it is the second sequence metadata statement, raise an error.

# CheckAssignSyntax

For each assignment statement:
1. If the lhs is not an expression which could refer to a section of the local variable array, raise an error.
2. If there is a type annotation, and the lhs is not an unqualified identifier, raise an error.

For each function definition statement:
1. If a parameter without a default value follows a parameter with a default value, raise an error.

For each sequence metadata statement:
1. If the number of parameters is greater than 255, raise an error.

# DefineFunctions

For each function definition statement:
1. The function identifier I is the name of the function F.
2. If I is already assigned in the enclosing scope S of the function definition statement, raise an error.
3. I is assigned to F in S.

# DefineVariables

This is split into two phases. Both phases visit statements recursively in breadth-first order. The first phase starts from the global scope and visits all statements, except those in function bodies. The second phase visits only statements inside of function bodies.

For each statement:

If the statement is an assignment statement:
1. If the assignment doesn't have a type annotation, skip it.
2. Otherwise, the lhs I must be an unqualified identifier by CheckAssignSyntax.
3. Create variable V of a type to be determined later.
4. I is the name of V.
5. If I is already assigned in the enclosing scope S of the assignment statement, raise an error.
6. I is assigned to V in S.

If the statement is a for loop statement:
1. Create the loop variable V of LoopVarType.
2. The loop var identifier I is the name of the loop variable V.
3. I must be undefined in the scope inside the for loop, as the statements in that scope have yet to be visited.
4. I is assigned to V in the scope inside the for loop.
5. Create the anonymous upper bound variable U of LoopVarType.
6. U is mapped to the for loop statement.

If the statement is a function definition statement:
1. For each function parameter P_i = (ident_i, type_i, default_i):
  1. Create variable V_i of a type to be determined later.
  2. The identifier ident_i is the name of V_i.
  2. If the identifier ident_i is already assigned in the scope S inside the function definition statement, raise an error.
  4. ident_i is assigned to V_i in S.

If the statement is a sequence metadata statement:
1. For each sequence parameter P_i = (ident_i, type_i):
  1. Create variable V_i of a type to be determined later.
  2. The identifier ident_i is the name of V_i.
  2. If the identifier ident_i is already assigned in the enclosing scope S of the sequence metadata statement, raise an error.
  4. ident_i is assigned to V_i in S.


# CheckBreakAndContinueInLoop
Visiting statements recursively in breadth-first order. 

For each while or for loop statement L:
1. Let C be any break or continue statement in the scope inside L
2. The enclosing loop of C is mapped to L

For each break or continue statement C:
1. If no enclosing loop has been mapped to C, raise an error.

# CheckReturnInFunc
Visiting statements recursively in breadth-first order. 

For function definition statement D
1. Let R be any return statement in the scope inside D
2. The enclosing function of R is mapped to D

For each return statement R:
1. If no enclosing function has been mapped to R, raise an error.


# Segue on name groups

The qualified names of definitions reside in the following name groups:
* The value name group, consisting of definitions of:
  * Variables
  * Constants
  * Telemetry channels
  * Parameters
  * Enum constants
  * Modules
* The callable name group, consisting of definitions of:
  * Casts
  * Builtins
  * Commands
  * Type constructors
  * Functions
  * Modules
* The type name group, consisting of type definitions and module definitions.

# Segue on dictionary definitions

For each type definition T in the `typeDefinitions` section of the dictionary:

1. The `qualifiedName` string is assumed to consist of a series of period-separated identifiers I_0, I_1, ..., I_n, forming a qualified identifier I.
2. I is the name of T in the global scope.
3. I is also the name of the type constructor of T in the global scope.

> These do not conflict because they are in separate name groups.

<!---
# the actual specifier is "abstract", does not refer to one specific command, but rather a command
# which remains to be instantiated
# in the dictionary, the commands in the commands section are all "instantiated" commands. But they do
# not have individual definitions. But for the purpose of Fpy, we will pretend that they are individually
# defined--therefor
-->

For each entry in the `commands` section of the dictionary:

1. The entry is considered the definition of command C, as far as Fpy is concerned.
2. The `name` string is assumed to consist of a series of period-separated identifiers I_0, I_1, ..., I_n, forming a qualified identifier I.
3. I is the name of C in the global scope.

Likewise with `telemetryChannels` and `parameters`.

For each constant definition C in the `constants` section of the dictionary:

1. The `qualifiedName` string is assumed to consist of a series of period-separated identifiers I_0, I_1, ..., I_n, forming a qualified identifier I.
2. I is the name of C in the global scope.
3. TODO on how the type and value of C are parsed

In each of these dictionary definitions, if n > 0, and the qualified identifier Q formed by I_0, ..., I_x for any x < n has not been seen before in this name group, then the definition is also considered a definition of module M in this name group, and Q is the name of M.

# ResolveIdentifiers

A qualified identifier is one of the following:

* An identifier.
* Q `.` I, where Q is a qualified identifier and I is an identifier.

To resolve a qualified identifier Q with a definition in name group N (i.e. associate Q with a definition):
1. Let R be the leftmost identifier of Q (if Q is an identifier, R is Q; otherwise Q is of the form Q' `.` I and R is the leftmost identifier of Q').
2. Resolve the identifier R in name group N (see below).
3. If R could not be resolved, raise an error.
4. Walk the remaining identifiers of Q from leftmost to rightmost. For each Q_i `.` I_i where Q_i has been resolved:
   1. If Q_i refers to a symbol which may contain sub definition, look up I_i within that symbol. If not found, raise an error. Otherwise, Q_i `.` I_i refers to the looked-up definition.
   2. Otherwise, Q_i refers to a definition which may not contain sub definitions, so no resolution is possible. However, it may still be a member access expression, so break out of this loop without an error.
   

To resolve an identifier I with a definition in name group N:
1. Let S be the enclosing scope of I.
2. If a definition D with name I in name group N exists in S, I refers to D.
3. Otherwise, if S has no parent P, raise an error.
4. Otherwise, go to step 2, but replace S with P.

For each expression or statement N:

If N is a function definition statement:
1. Resolve the function identifier in the callable name group.
2. Resolve the return type expression, if present, in the type name group.
3. For each parameter P_i = (ident_i, type_i, default_i):
   1. ident_i refers to the parameter variable defined in DefineVariables.
   2. Resolve type_i in the type name group.
   3. Resolve default_i, if present, in the value name group.

If N is an assignment statement:
1. Resolve the type annotation, if present, in the type name group.
2. Resolve the lhs in the value name group.
3. Resolve the rhs in the value name group.

If N is a sequence metadata statement:
1. For each parameter P_i = (ident_i, type_i):
   1. ident_i refers to the parameter variable defined in DefineVariables.
   2. Resolve type_i in the type name group.

If N is a function call expression:
1. Resolve the callable expression in the callable name group.
2. For each argument: resolve the argument expression (or, for named arguments, the argument's value expression) in the value name group.

If N is the condition of an if, elif, while, or assert statement, resolve the condition in the value name group. For an assert statement, also resolve the exit code expression, if present, in the value name group.

If N is a binary or unary operator expression, resolve each operand in the value name group.

If N is a for loop statement:
1. The loop variable identifier refers to the variable defined in DefineVariables.
2. Resolve the range expression in the value name group.

If N is an index expression:
1. Resolve the parent expression in the value name group.
2. Resolve the index expression in the value name group.

If N is a range expression, resolve the lower and upper bound expressions in the value name group.

If N is a return statement, resolve the return value expression, if present, in the value name group.

If N is an anonymous struct expression, resolve each member's value expression in the value name group.

If N is an anonymous array expression, resolve each element expression in the value name group.

Otherwise, skip N.


# CheckAllUnqualifiedIdentifiersResolved

For each unqualified identifier I, if I has not been resolved to a definition, raise an error.

# CheckAllTypesAndCallablesResolved

For each expression or statement N:

If N is a function definition statement:
1. Check that the function identifier resolved to a callable.
2. Check that the return type expression, if present, resolved to a type.
3. For each parameter P_i = (ident_i, type_i, default_i):
   1. Check that type_i resolved to a type.

If N is an assignment statement, check that the type annotation, if present, resolved to a type.

If N is a sequence metadata statement:
1. For each parameter P_i = (ident_i, type_i):
   2. Check that type_i resolved to a type.

If N is a function call statement, check that the callable expression resolved to a callable.

Otherwise, skip N.

If any check fails, raise an error.

> At this point, either all uses of definitions have been resolved, or an error has been raised.

# CheckForConstantSizeTypes

For each expression or statement N:

If N is a function definition statement:
2. Check that the return type expression, if present, is a constant sized type.
3. For each parameter P_i = (ident_i, type_i, default_i):
   1. Check that type_i resolved to a type.

Otherwise, skip N.