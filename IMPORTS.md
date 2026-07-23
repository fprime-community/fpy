An **import statement** makes another sequence's [function definitions](#function-definition) available in the importing sequence.

## Syntax

Rule:

`import_stmt: import_direct | import_from`
`import_direct: "import" "."* name ("." name)* ["as" name]`
`import_from: "from" "."* name ("." name)* "import" ("*" | import_members | "(" import_members [","] ")")`
`import_members: name ["as" name] ("," name ["as" name])*`

Name:

`import_direct: "import" [dots] import_path ["as" alias]`
`import_from: "from" [dots] import_path "import" ("*" | members | "(" members [","] ")")`
`members: member ["as" alias] ("," member ["as" alias])*`
`dots: "."+`

In the parenthesized form, the member list may span multiple lines.

An import statement is only valid outside an indentation block.

An import statement with one or more leading dots is a **relative import statement**, otherwise it is an **absolute import statement**.

> Unlike Python, `import .util` is valid and `from . import util` is not.

If the `import_from` syntax is used, the import statement is an **import-from statement**. If the `*` syntax is used in an import-from statement, it is an **import-star statement**.

If the `import_direct` syntax is used, it is a **direct import statement**.

## Semantics

### Constructing the AST

Let the **main sequence** refer to the sequence defined by the input file the user passes into the compiler.

For each import statement in the AST, including statements added by this process:

1. The import path must [resolve](#import-path-resolution) to an imported sequence file F, otherwise an error is raised.

2. If F has previously been included in the program's AST, or if it is the main sequence, skip it.

3. Otherwise, F is lexed and parsed according to this specification, producing a new block B.

4. If B has top-level statements which may have side effects, an error is raised.

5. If B has a sequence metadata statement with one or more formal parameters, an error is raised.

6. B is included in the program's AST as a sibling of the main sequence's block.

> Cyclical imports are allowed. This is not an issue because import statements cannot have side effects.

#### Import path resolution

Import path resolution is the process by which the qualified identifier `import_path` is resolved to a sequence file.

Each file in the filesystem whose name (without a preceding path) is of the form `<name>.fpy` is a **sequence file**, associated with name `name`.

The **import directories** are an ordered list of absolute paths of directories provided by the environment in which the compiler is invoked.
> In the command-line compiler, the import directories are passed with `-i`/`--imports`.

Relative import statements have an **anchor directory**, which is the Nth parent directory of the absolute path of the sequence file containing the statement, where N is the number of dots preceding `import_path`. If the sequence was not read from a file, or if there is no Nth parent directory, an error is raised.

In a directory D, an identifier I refers to the sequence file or directory in D named I, if one exists. If D contains both, an error is raised.

If the import statement is an absolute import statement, resolution of I is attempted in each import directory in order until it succeeds. If I cannot be resolved in any import directory, an error is raised.

If the import statement is a relative import statement, resolution of I is attempted in the anchor directory. An error is raised if I cannot be resolved.

To resolve qualified identifier Q.I:
1. Recursively resolve Q.
2. If Q resolves to a sequence file: if the import statement is an import-from statement, an error is raised. Otherwise, I refers to the definition with name I in Q; if none exists, an error is raised.
3. Otherwise, if Q resolves to a directory, resolution of I is attempted in Q. If I could not be resolved, an error is raised.
4. Otherwise, Q resolved to neither a directory nor a sequence file, and an error is raised.

These rules are applied to `import_path`. If it refers to a directory, an error is raised. Otherwise, it refers to a sequence file, or to a definition in one; that file is the **imported sequence file** F.

### Binding

The **importing sequence** is the sequence containing the import statement; the **importing scope** is its scope.

An import statement associates one or more qualified names with definitions in the importing scope.

The **imported scope** is the scope of the sequence defined by the imported sequence file.

For an import-star statement:
For each name N which refers to a definition D in the imported scope:
1. If N begins with an underscore, skip it.
2. Otherwise, associate N with D in the importing scope.

For other import-from statements:
For each member with name N and optional alias A in the `members` list:
1. If N is not associated with a definition in the imported scope, an error is raised.
2. Otherwise, let D be the definition associated with N.
3. If the optional alias A is provided, associate A with D in the importing scope.
4. Otherwise, associate N with D in the importing scope.

Otherwise, the import statement is a direct import statement:
1. If `import_path` refers to a sequence file, let D be the sequence defined by that file.
2. Otherwise, `import_path` refers to a definition D in a sequence file.
3. If the optional alias A is provided, A is associated with D in the importing scope.
4. Otherwise, `import_path` is the qualified name of D in the importing sequence.

