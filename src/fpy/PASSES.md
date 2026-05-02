# CreateScopes

A scope is an ordered list of statements at a single, continuous indentation level.

If a scope is the global scope, it has no parent scope.
Otherwise, the parent scope of scope S is the scope at one indentation level lower than S.

If the scope is part of a function definition, it is a function scope.

To determine the enclosing scope of an expression E:
# TODO remove the special casing of func prms/loop vars
3. Otherwise, the enclosing scope of E is the nearest parent scope of E.

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
1. The function identifier becomes the name N of the function F.
2. If N is already assigned in the enclosing scope S of the function definition statement, raise an error.
3. N is assigned to F in S.

# DefineVariables

This is split into two phases. Both phases visit statements recursively in breadth-first order. The first phase starts from the global scope and visits all statements, except those in function bodies. The second phase visits only statements inside of function bodies.

For each statement:

If the statement is an assignment statement:
1. If the assignment doesn't have a type annotation, skip it.
2. Otherwise, the lhs must be an unqualified identifier by CheckAssignSyntax.
3. Create variable V of a type to be determined later.
4. The lhs is the name N of V.
5. If N is already assigned in the enclosing scope S of the assignment statement, raise an error.
6. N is assigned to V in S.

If the statement is a for loop statement:
1. Create the loop variable V of LoopVarType.
2. The loop var identifier is the name N of the loop variable V.
3. N must be undefined in the scope inside the for loop, as the statements in that scope have yet to be visited.
4. N is assigned to V in the scope inside the for loop.
5. Create the anonymous upper bound variable U of LoopVarType.
6. U is mapped to the for loop statement.

If the statement is a function definition statement:
1. For each function parameter P_i = (ident_i, type_i, default_i):
  1. Create variable V_i of a type to be determined later.
  2. The identifier ident_i is the name N_i of V_i.
  2. If the identifier ident_i is already assigned in the scope S inside the function definition statement, raise an error.
  4. N_i is assigned to V_i in S.

If the statement is a sequence metadata statement:
1. For each sequence parameter P_i = (ident_i, type_i):
  1. Create variable V_i of a type to be determined later.
  2. The identifier ident_i is the name N_i of V_i.
  2. If the identifier ident_i is already assigned in the enclosing scope S of the sequence metadata statement, raise an error.
  4. N_i is assigned to V_i in S.


# CheckBreakAndContinueInLoop
Visiting statements recursively in breadth-first order. 

For each while or for loop statement L:
1. Let C be any break or continue statement in the scope inside L
2. The enclosing loop of C is mapped to L

For each break or continue statement C:
1. If no enclosing loop has been mapped to C, raise an error.

# Segue on name groups

The qualified names of definitions reside in the following name groups:
* The value name group, consisting of definitions of:
  * Variables
  * Constants
  * Telemetry channels
  * Parameters
  * Enum constants
* The callable name group, consisting of definitions of:
  * Casts
  * Builtins
  * Commands
  * Type constructors
  * Functions
* The type name group, consisting of type definitions.

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

In each of these dictionary definitions, if n > 0, and the qualified identifier Q formed by I_0, ..., I_x for any x < n has not been seen before in this name group, then the definition is also considered a definition of namespace N in this name group, and Q is the name of N.

# ResolveTypeAndCallableUses

A qualified identifier is one of the following:

* An identifier.
* Q `.` I, where Q is a qualified identifier and I is an identifier.

To associate an unqualified identifier with a definition (a process called resolution), 

To associate a qualified identifier with a definition (a process called resolution), 

<!---
an important diff between Fpp and Fpy is that a definition cannot have sub definitions.
Basically, each definition is terminal.

If you think about it, from the dictionary perspective again this makes sense. There's no 
dictionary definition of modules or components. So those qualifiers really aren't
referring to any definitions. I can't recover the definitions from the strings.
-->