# CreateScopes

A scope is an ordered list of statements at a single, continuous indentation level.

If a scope is the global scope, it has no parent scope.
Otherwise, the parent scope of scope S is the scope at one indentation level lower than S.

If the scope is part of a function definition, it is a function scope.

To determine the enclosing scope of an expression E:
1. If E is the loop variable of a for loop, its enclosing scope is the scope inside the for loop.
2. If E is a parameter identifier of a function, its enclosing scope is the scope inside the function definition.
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
2. N is associated with F in the enclosing scope of the function definition statement.

# DefineVariables

This is split into two phases. Both phases visit statements recursively in breadth-first order. The first phase starts from the global scope and visits all statements, except those in function bodies. The second phase visits only statements inside of function bodies.

For each statement:

If the statement is an assignment statement:
1. If the assignment doesn't have a type annotation, skip it.
2. Otherwise, the lhs must be an unqualified identifier by CheckAssignSyntax.
3. The lhs becomes the name N of the variable V.
4. If N is already associated with a definition in the enclosing scope S of the assignment statement, raise an error.
5. N is associated with V in S.

If the statement is a for loop statement:
1. The loop var identifier becomes the name N of the loop variable V.
2. N is associated with V in the scope inside the for loop.
3. An anonymous variable U is defined to be the for loop upper bound variable.