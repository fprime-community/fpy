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
1. For each parameter P_i = (ident_i, type_i, default_i):
  1. Create variable V_i of a type to be determined later.
  2. The identifier ident_i is the name N_i of V_i.
  2. If the identifier ident_i is already assigned in the scope S inside the function definition statement, raise an error.
  4. N_i is assigned to V_i in S.

If the statement is a sequence metadata statement:
1. For each parameter P_i = (ident_i, type_i, default_i):
  1. Create variable V_i of a type to be determined later.
  2. The identifier ident_i is the name N_i of V_i.
  2. If the identifier ident_i is already assigned in the enclosing scope S of the sequence metadata statement, raise an error.
  4. N_i is assigned to V_i in S.


# CheckBreakAndContinueInLoop
