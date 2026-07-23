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

A line of the specification may be followed by an italicized *Tests:* line. Each bracketed number links to a test that verifies the behavior; hover over a number to see the test's full `file::class::name`. These links are checked and kept up to date by `verify/spec_links.py`.

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
* `import`
* `not`
* `pass`
* `return`
* `while`

## Symbols

A **symbol** is a language construct that can be referred to by a name in the program. 

The following language constructs may be symbols:
* [modules](#modules)
* [variables](#variables)
* [callables](#callables)
* [types](#types)
* telemetry channels
* parameters
* enum constants
TODO members?
TODO you're selecting a member of a value--not referring to a static member. when you say a.b you're asking for a comp to be performed

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

Each name group only contains names which map to their particular language construct, or modules with the same property, recursively.
TODO Maybe could remove this line

TODO explain why we need this
Name groups do not intersect.

> This means that the names of callables, types and values never conflict.  WITH EACH OTHER

Name groups are accessed via syntactic context.
TODO this is really just an explanation

TODO type names should NOT be expressions


TODO A.b is an expr, A is an expr, b is not an expression. A.b is a dot expression. b is not an expression, it's part of a 
TODO Does it refer to something that has a scope, or does it refer to something that has members.

> For instance, the type name group is accessible anywhere in the source code where a type name is expected, such as a [variable definition](#variable-definition) type annotation, or a [function definition](#function-definition) return type.

The **resolving name group** is the name group that a name should be resolved in, based on its syntactic context.

## Shadowing

A [definition](#definitions) **shadows** an outer name when it introduces a name into its [resolving scope](#scopes)'s [name group](#name-groups) that is *not* already defined in that scope, but *does* resolve to a symbol in an enclosing scope's same name group.

Each shadowing definition emits a warning named for the name group it shadows:
* Shadowing a name in the [value name group](#name-groups) emits the `shadow-value` warning.
* Shadowing a name in the [callable name group](#name-groups) emits the `shadow-callable` warning.

*Tests:* [1](test/fpy/test_variables.py#L241 "test/fpy/test_variables.py::TestShadowWarnings::test_inner_block_shadows_outer_variable_warns"), [2](test/fpy/test_variables.py#L255 "test/fpy/test_variables.py::TestShadowWarnings::test_root_variable_shadows_dictionary_name_warns"), [3](test/fpy/test_functions.py#L932 "test/fpy/test_functions.py::TestShadowWarnings::test_function_shadows_builtin_warns"), [4](test/fpy/test_imports.py#L799 "test/fpy/test_imports.py::TestImportShadowsBuiltins::test_import_shadowing_builtin_warns")

## Modules

A **module** is a mapping of names to symbols, associated with a name.

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
    3. If the qualifier is not a module, an error is raised.
    4. Resolve the name in the qualifier module.

If at any point a name fails to be resolved, an error is raised, unless otherwise specified.

A **fully-qualified name** is a qualified name which is not itself a qualifier.
TODO names are semantic, ident is syntactic
TODO you can't actually tell at syntax level what is a fqn
If a fully-qualified name resolves to a module, an error is raised.

*Tests:* [1](test/fpy/test_imports.py#L552 "test/fpy/test_imports.py::TestImportIsolation::test_sequence_name_not_usable_as_value")

TODO you can think of the dict as "importing definitions"


> Module symbols cannot be used anywhere, so this forces names to resolve to something "useful"

TODO I'm not sure this is clear what this means. The idea here is that the full qualified name should always reference SOMETHING--cannot just put Svc in place of a type, even though Svc does resolve in type name group and global scope.

## Definitions

A **definition** is a language construct that introduces a name-to-[symbol](#symbols) mapping as a part of a [scope](#scopes) and [name group](#name-groups).

The list of definitions is:
* [Variable definitions](#variable-definition)
* [Function definitions](#function-definition)
* Module, sequence, and alias definitions, made by [import statements](#imports)

### Semantic equivalence

Two syntactic definitions may be **semantically equivalent**. The following rules lay out when two definitions are considered semantically equivalent:

* Two variable definitions are never semantically equivalent.
* Two function definitions are never semantically equivalent.
* Two module definitions with the same qualified name are always semantically equivalent.
* Two sequence definitions are semantically equivalent iff they name the same sequence file.
* Two alias definitions are semantically equivalent iff they name the same definition.
* Two definitions of different kinds are never semantically equivalent.

If two definitions that are not semantically equivalent introduce the same qualified name into the same [scope](#scopes) and a shared [name group](#name-groups), it is a **collision**, and an error is raised.

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

If `lhs` is already defined in the resolving scope's [value name group](#name-groups), an error is raised.

> This prevents redefining a variable in the same scope.

If `lhs` is not defined in the resolving scope, but resolves to a symbol in an enclosing scope's value name group, the definition [shadows](#shadowing) that outer name, and emits the `shadow-value` warning.

*Tests:* [1](test/fpy/test_variables.py#L241 "test/fpy/test_variables.py::TestShadowWarnings::test_inner_block_shadows_outer_variable_warns"), [2](test/fpy/test_variables.py#L255 "test/fpy/test_variables.py::TestShadowWarnings::test_root_variable_shadows_dictionary_name_warns"), [3](test/fpy/test_variables.py#L266 "test/fpy/test_variables.py::TestShadowWarnings::test_same_scope_variable_redeclare_is_error")

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

If `name` is already defined in the resolving scope's [callable name group](#name-groups), an error is raised.

> This prevents redefining a function in the same scope.

If `name` is not defined in the resolving scope, but resolves to a callable in an enclosing scope's callable name group, the definition [shadows](#shadowing) that outer callable, and emits the `shadow-callable` warning.

*Tests:* [1](test/fpy/test_functions.py#L932 "test/fpy/test_functions.py::TestShadowWarnings::test_function_shadows_builtin_warns"), [2](test/fpy/test_functions.py#L39 "test/fpy/test_functions.py::TestDefinition::test_redeclare_func")

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

`check_stmt: "check" expr check_clause* ":" stmt_list ["timeout" ":" stmt_list]`

`check_clause: "timeout" expr | "persist" expr | "freq" expr`

The clauses can appear in any order, and can be spread across multiple indented lines (with the colon after the last clause).

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

# Imports

An **import statement** compiles another Fpy sequence, and makes the sequence's [definitions](#definitions) available in the importing sequence.

## Syntax

Rule:

`import_stmt: import_seq | import_from`
`import_seq: "import" "."* name ("." name)* ["as" name]`
`import_from: "from" "."* name ("." name)* "import" ("*" | import_members | "(" import_members [","] ")")`
`import_members: name ["as" name] ("," name ["as" name])*`

Name:

`import_seq: "import" [dots] import_path ["as" alias]`
`import_from: "from" [dots] sequence_path "import" ("*" | members | "(" members [","] ")")`
`members: member ["as" alias] ("," member ["as" alias])*`

The **import path** is the dotted chain of names, excluding any leading dots. Its first name is its **root segment**, and its last is its **leaf segment**. The **alias** is the name introduced by an `as` clause.

An import statement with one or more leading dots is **relative**; one with none is **absolute**. The leading dots must be followed by an import path of at least one name.

*Tests:* [1](test/fpy/test_imports.py#L2329 "test/fpy/test_imports.py::TestImportRelative::test_bare_dot_from_is_error")

> Unlike Python, `import .util` is valid and `from . import util` is not. The members of a `from` statement are always definitions, never sequences; to import a sibling sequence whole, write `import .util`.

In the parenthesized form, the member list may span multiple lines.

An import statement is only valid outside an indentation block.

*Tests:* [1](test/fpy/test_imports.py#L937 "test/fpy/test_imports.py::TestImportOnlyAtTopLevel::test_import_inside_if_block_fails"), [2](test/fpy/test_imports.py#L954 "test/fpy/test_imports.py::TestImportOnlyAtTopLevel::test_import_inside_function_fails")

## Semantics

### Sequence resolution

The **base import search path** is a list of directories provided by the environment in which the compiler is invoked. It is the same for every sequence in a compilation.

> In the command-line compiler, the base import search path is each directory passed with `-i`/`--imports`; exact duplicates (after path resolution) are dropped, so a repeated `-i` flag cannot manufacture an ambiguity error. The main sequence's own directory is not implicitly added: a sequence reaches its own siblings with relative imports.

An import statement is resolved against **candidate directories**:
* The candidate directories of an absolute import are the base import search path. The location of the importing sequence plays no role, so an absolute import path names the same file in every sequence of a compilation.
* A relative import has a single candidate directory, its **anchor**: one leading dot anchors at the directory containing the importing sequence, and each additional leading dot moves the anchor to its parent directory. The base import search path plays no role.

*Tests:* [1](test/fpy/test_imports.py#L2173 "test/fpy/test_imports.py::TestImportRelative::test_relative_import_of_sibling"), [2](test/fpy/test_imports.py#L2196 "test/fpy/test_imports.py::TestImportRelative::test_absolute_import_ignores_importer_directory"), [3](test/fpy/test_imports.py#L2222 "test/fpy/test_imports.py::TestImportRelative::test_relative_import_does_not_search_base_path"), [4](test/fpy/test_imports.py#L2265 "test/fpy/test_imports.py::TestImportRelative::test_parent_relative_import")

If a relative import appears in a sequence that has no containing directory (such as one compiled from a stream), an error is raised.

*Tests:* [1](test/fpy/test_imports.py#L2414 "test/fpy/test_imports.py::TestImportRelative::test_relative_import_without_location_is_error")

A sequence path `s_0.s_1. ... .s_n` **resolves** in a directory `dir` if the file `dir/s_0/s_1/.../s_n.fpy` exists.

*Tests:* [1](test/fpy/test_imports.py#L1210 "test/fpy/test_imports.py::TestImportDottedPaths::test_single_dotted_import"), [2](test/fpy/test_imports.py#L1229 "test/fpy/test_imports.py::TestImportDottedPaths::test_deeply_nested_dotted_import")

An import statement names its imported sequence and an optional member as follows. Let its import path be `s_0. ... .s_n`. In each candidate directory:
1. If `s_0. ... .s_n` resolves in the directory, that file is the directory's **split**; the whole path is the **sequence path**, and there is no member.
2. Otherwise, if `n > 0` and `s_0. ... .s_{n-1}` resolves in the directory, that file is the directory's split; `s_0. ... .s_{n-1}` is the sequence path, and the final name `s_n` is the **member**.
3. Otherwise, the directory has no split.

If exactly one candidate directory has a split, it names the imported sequence and the member. If no candidate directory has one, an error is raised. If more than one candidate directory has one, an error is raised, even if the splits name the same file.

A `from` statement's path splits by step 1 only: it is always the whole sequence path, never a sequence path plus member. Each name in its import list is a member.

*Tests:* [1](test/fpy/test_imports.py#L1372 "test/fpy/test_imports.py::TestImportDirectories::test_sequence_found_in_later_import_directory"), [2](test/fpy/test_imports.py#L1430 "test/fpy/test_imports.py::TestImportDirectories::test_dotted_sequence_resolved_across_import_directories"), [3](test/fpy/test_imports.py#L349 "test/fpy/test_imports.py::TestImportErrors::test_missing_sequence_is_an_error"), [4](test/fpy/test_imports.py#L1448 "test/fpy/test_imports.py::TestImportDirectories::test_no_import_directories_cannot_resolve"), [5](test/fpy/test_imports.py#L1267 "test/fpy/test_imports.py::TestImportDottedPaths::test_missing_leaf_in_existing_directory_is_error"), [6](test/fpy/test_imports.py#L1341 "test/fpy/test_imports.py::TestImportFileDirectoryConflict::test_bare_directory_import_is_error"), [7](test/fpy/test_imports.py#L1354 "test/fpy/test_imports.py::TestImportFileDirectoryConflict::test_dotted_leaf_directory_import_is_error"), [8](test/fpy/test_imports.py#L507 "test/fpy/test_imports.py::TestImportFileErrors::test_import_path_is_a_directory_fails")

> A file `foo.fpy` always takes precedence over a sibling directory `foo/` for `import foo`, while `import foo.bar` descends into the directory `foo/` regardless of whether `foo.fpy` exists.

> Splitting `import a.b.c` against a directory:
> * `a/b/c.fpy` exists: sequence path `a.b.c`, no member -- the whole sequence is imported.
> * `a/b/c.fpy` is missing but `a/b.fpy` exists: sequence path `a.b`, member `c`.
> * neither exists: the directory has no split.
>
> The whole path is preferred over a member within a single directory: if both `a/b/c.fpy` and `a/b.fpy` exist in one directory, `import a.b.c` imports `a/b/c.fpy` whole. Nothing is preferred across directories: two base directories that both contain a `util.fpy` make `import util` ambiguous. Ambiguity is an error rather than a shadowing rule so that adding a file to a search directory can never silently change which file another import names.

> The intended layout for a reusable library is a directory of sequences that reference one another with relative imports. A consumer adds the directory *containing* the library directory to the base import search path, and writes `import <library>.<sequence>`. The library then works unmodified wherever it is checked out or copied to, and every consumer names its sequences identically.

### Importing a sequence

If the imported sequence fails to parse or compile, an error is raised.

*Tests:* [1](test/fpy/test_imports.py#L489 "test/fpy/test_imports.py::TestImportFileErrors::test_parse_error_in_imported_file_fails")

> The diagnostic should point into the imported file, not at the import statement.

If the imported sequence declares one or more [sequence arguments](todo), an error is raised.

*Tests:* [1](test/fpy/test_imports.py#L328 "test/fpy/test_imports.py::TestImportErrors::test_cannot_import_sequence_with_arguments")

> A `sequence()` directive with no arguments does not prevent a file from being imported.

*Tests:* [1](test/fpy/test_imports.py#L357 "test/fpy/test_imports.py::TestImportErrors::test_no_arg_sequence_is_importable")

The imported sequence is compiled as a sequence in its own right: names in it resolve in its own, new global scope, the rules governing a sequence's own statements apply to it unchanged, and the semantics of this section apply recursively to its own import statements, with the imported sequence in the role of the importing sequence.

*Tests:* [1](test/fpy/test_imports.py#L976 "test/fpy/test_imports.py::TestImportTransitive::test_transitive_import_works"), [2](test/fpy/test_imports.py#L607 "test/fpy/test_imports.py::TestImportIsolation::test_imported_function_cannot_see_importer_globals"), [3](test/fpy/test_imports.py#L1004 "test/fpy/test_imports.py::TestImportTransitive::test_transitive_dependency_is_private"), [4](test/fpy/test_imports.py#L378 "test/fpy/test_imports.py::TestImportErrors::test_imported_sequence_metadata_must_be_first")

> Isolation is by construction. The importing sequence's symbols live in the importing sequence's global scope, so they are not visible in the imported sequence; a sequence imported by the imported sequence binds names in the imported sequence's global scope, so it is not visible in the importing sequence. Every rule below that binds a name says which global scope receives it.

A sequence may transitively import itself. Such an import is subject to the rules of this section like any other, and the sequence is compiled once, like any other.

*Tests:* [1](test/fpy/test_imports.py#L1040 "test/fpy/test_imports.py::TestImportCycles::test_self_import"), [2](test/fpy/test_imports.py#L1064 "test/fpy/test_imports.py::TestImportCycles::test_mutual_import"), [3](test/fpy/test_imports.py#L1103 "test/fpy/test_imports.py::TestImportCycles::test_three_way_cycle"), [4](test/fpy/test_imports.py#L1143 "test/fpy/test_imports.py::TestImportCycles::test_cycle_compiles_each_sequence_once")

An import statement introduces names as [definitions](#definitions) in the importing sequence's global scope, of three kinds:
* A **module definition** defines a [module](#modules).
* A **sequence definition** defines a **sequence symbol**: a module that holds (some of) an imported sequence's definitions and names the file they come from.
* An **alias definition** defines an existing definition again under a new name.


An import statement with no member and no alias makes a module definition at each proper, non-empty prefix of its sequence path, and a sequence definition at the full path, holding each definition in the imported sequence's global scope under its own name.

*Tests:* [1](test/fpy/test_imports.py#L111 "test/fpy/test_imports.py::TestImportInlining::test_call_imported_function"), [2](test/fpy/test_imports.py#L132 "test/fpy/test_imports.py::TestImportInlining::test_local_and_imported_names_coexist"), [3](test/fpy/test_imports.py#L533 "test/fpy/test_imports.py::TestImportIsolation::test_imported_symbol_requires_qualification"), [4](test/fpy/test_imports.py#L577 "test/fpy/test_imports.py::TestImportIsolation::test_same_function_name_in_two_sequences_no_collision"), [5](test/fpy/test_imports.py#L1247 "test/fpy/test_imports.py::TestImportDottedPaths::test_dotted_symbol_requires_full_path")

> After `import a.b.c` of a sequence that defines `x`, the symbol is named `a.b.c.x`, and is available under no shorter name.

The sequence path of a relative import excludes its leading dots; the dots affect resolution only.

*Tests:* [1](test/fpy/test_imports.py#L2243 "test/fpy/test_imports.py::TestImportRelative::test_relative_binds_path_after_dots"), [2](test/fpy/test_imports.py#L2265 "test/fpy/test_imports.py::TestImportRelative::test_parent_relative_import")

> `import .sub.mod` binds `sub.mod`, and `import ..util` binds `util`. So the depth of the importer never leaks into names: `lib/a.fpy` (via `import .util`) and `lib/sub/b.fpy` (via `import ..util`) both bind `lib/util.fpy` as `util`.

The conflict rule spans everything in the scope: an import-introduced name also conflicts with a non-import [definition](#definitions) (a function or variable) of the same qualified name in a shared name group. A module or sequence symbol resides in the [name groups](#name-groups) of the definitions it (transitively) holds; a symbol holding no definitions resides in no name group and binds nothing.

A collision is only with a name in the importing scope itself (or the same module). If the imported name is instead free in the importing scope but resolves to a symbol in an enclosing scope, the import [shadows](#shadowing) that name rather than colliding. This emits the `shadow-callable` warning if the bound name occupies the callable name group, or the `shadow-value` warning if it occupies the value name group.

*Tests:* [1](test/fpy/test_imports.py#L799 "test/fpy/test_imports.py::TestImportShadowsBuiltins::test_import_shadowing_builtin_warns"), [2](test/fpy/test_imports.py#L820 "test/fpy/test_imports.py::TestImportShadowsBuiltins::test_from_import_alias_shadowing_builtin_warns"), [3](test/fpy/test_imports.py#L838 "test/fpy/test_imports.py::TestImportShadowsBuiltins::test_imported_sequence_function_shadowing_builtin_warns")

*Tests:* [1](test/fpy/test_imports.py#L636 "test/fpy/test_imports.py::TestImportNameCollisions::test_import_collides_with_local_function"), [2](test/fpy/test_imports.py#L657 "test/fpy/test_imports.py::TestImportNameCollisions::test_import_collides_with_local_variable"), [3](test/fpy/test_imports.py#L676 "test/fpy/test_imports.py::TestImportNameCollisions::test_import_coexists_with_local_variable"), [4](test/fpy/test_imports.py#L1641 "test/fpy/test_imports.py::TestImportAlias::test_alias_collides_with_local"), [5](test/fpy/test_imports.py#L1988 "test/fpy/test_imports.py::TestImportFrom::test_from_import_collides_with_local"), [6](test/fpy/test_imports.py#L2009 "test/fpy/test_imports.py::TestImportFrom::test_from_star_collides_across_sequences"), [7](test/fpy/test_imports.py#L1850 "test/fpy/test_imports.py::TestImportFrom::test_from_import_duplicate_member_warns"), [8](test/fpy/test_imports.py#L1286 "test/fpy/test_imports.py::TestImportDottedPaths::test_two_sequences_in_same_module_no_collision"), [9](test/fpy/test_imports.py#L2510 "test/fpy/test_imports.py::TestImportModuleMerging::test_directory_and_sequence_same_name_forbidden"), [10](test/fpy/test_imports.py#L2535 "test/fpy/test_imports.py::TestImportModuleMerging::test_two_aliases_same_name_collide"), [11](test/fpy/test_imports.py#L2551 "test/fpy/test_imports.py::TestImportModuleMerging::test_relative_and_absolute_leaf_collision"), [12](test/fpy/test_imports.py#L2454 "test/fpy/test_imports.py::TestImportModuleMerging::test_sibling_modules_merge")

> So after `import pkg.a` and `import pkg.b`, module `pkg` contains both `a` and `b`. And a variable `lib` coexists with a functions-only imported sequence symbol `lib`, because a value and a callable never share a name group.

> Two DIFFERENT files never pool under one name: an absolute `import util` and a relative `import .util` that name different files collide -- rename one with `as`. The SAME file imported more than one way instead folds together idempotently. And `import pkg` (the file `pkg.fpy`) together with `import pkg.mod` is the forbidden clash of one name used as both a sequence and a module.

An import statement with a member and no alias makes a module definition at each proper, non-empty prefix of its sequence path, and a sequence definition at the full path holding only the [definition](#definitions) named by the member in the imported sequence's global scope, under the member's name. If the imported sequence's global scope has no such definition, an error is raised.


> `import a.b.c.foo` adds only `foo`, named `a.b.c.foo`.

An import statement with an alias introduces no modules. With a member, it makes an alias definition binding the imported [definition](#definitions) under the alias. Without one, it makes a sequence definition at the alias, holding each definition in the imported sequence's global scope. The alias occupies the [name groups](#name-groups) of what it is bound to.

*Tests:* [1](test/fpy/test_imports.py#L1584 "test/fpy/test_imports.py::TestImportAlias::test_import_sequence_as_alias"), [2](test/fpy/test_imports.py#L1603 "test/fpy/test_imports.py::TestImportAlias::test_dotted_import_as_alias"), [3](test/fpy/test_imports.py#L1621 "test/fpy/test_imports.py::TestImportAlias::test_alias_hides_chain")

> `import a.b.c as x` binds `x` to a sequence symbol of `a.b.c`'s definitions, so writing it twice folds (one name, one file); `import a.b.c.foo as x` binds `x` to the definition `foo`.

A `from` statement introduces no modules. It makes an alias definition for each member: the [definition](#definitions) the member names in the imported sequence's global scope, under the member's name or its alias if one is given. `from p import *` makes an alias definition for each definition in the imported sequence's global scope whose name does not begin with an underscore, under its own name. If a member names no definition in the imported sequence's global scope, an error is raised.

If one member list binds the same name to the same definition more than once, the `import-duplicate` warning is emitted.

*Tests:* [1](test/fpy/test_imports.py#L1850 "test/fpy/test_imports.py::TestImportFrom::test_from_import_duplicate_member_warns"), [2](test/fpy/test_imports.py#L1910 "test/fpy/test_imports.py::TestImportFrom::test_from_import_one_name_two_members_is_error")

*Tests:* [1](test/fpy/test_imports.py#L1667 "test/fpy/test_imports.py::TestImportFrom::test_from_import_function"), [2](test/fpy/test_imports.py#L1700 "test/fpy/test_imports.py::TestImportFrom::test_from_dotted_sequence"), [3](test/fpy/test_imports.py#L1718 "test/fpy/test_imports.py::TestImportFrom::test_from_import_as_alias"), [4](test/fpy/test_imports.py#L1736 "test/fpy/test_imports.py::TestImportFrom::test_from_import_star"), [5](test/fpy/test_imports.py#L1934 "test/fpy/test_imports.py::TestImportFrom::test_from_import_does_not_introduce_sequence_name"), [6](test/fpy/test_imports.py#L1956 "test/fpy/test_imports.py::TestImportFrom::test_from_import_missing_member_is_error"), [7](test/fpy/test_imports.py#L1758 "test/fpy/test_imports.py::TestImportFrom::test_from_import_multiple_members"), [8](test/fpy/test_imports.py#L1779 "test/fpy/test_imports.py::TestImportFrom::test_from_import_multiple_with_aliases"), [9](test/fpy/test_imports.py#L1800 "test/fpy/test_imports.py::TestImportFrom::test_from_import_parenthesized_single_line"), [10](test/fpy/test_imports.py#L1821 "test/fpy/test_imports.py::TestImportFrom::test_from_import_parenthesized_multiline"), [11](test/fpy/test_imports.py#L248 "test/fpy/test_imports.py::TestImportUnderscore::test_star_import_does_not_bind_underscore_names"), [12](test/fpy/test_imports.py#L268 "test/fpy/test_imports.py::TestImportUnderscore::test_star_import_binds_public_names")

Two import statements import the same sequence if they resolve to the same file, whatever paths named it. 

A sequence is compiled once, however many import statements name it: in one importing sequence or across several, and whether named by one path or by different paths that resolve to the same file. Its definitions are shared, never duplicated.

> So a sequence and a sequence it imports may both import a third, and it is compiled once for the whole program.

*Tests:* [1](test/fpy/test_imports.py#L907 "test/fpy/test_imports.py::TestImportDuplicates::test_duplicate_across_files_is_allowed")

Importing the same sequence more than once in one sequence is governed by [semantic equivalence](#semantic-equivalence). Because a sequence names one file, two imports of it under one name are the same sequence definition and fold idempotently -- `import lib` twice, or `import lib.a` and `import lib.b`, are allowed. Likewise two aliases of one definition under one name fold idempotently -- `from lib import a` twice, or a definition re-exported by one sequence and imported again directly from another (a diamond). Binding a name held by a different definition still collides as usual.

*Tests:* [1](test/fpy/test_imports.py#L2102 "test/fpy/test_imports.py::TestImportDuplicateSequence::test_from_import_repeated_across_statements_folds"), [2](test/fpy/test_imports.py#L2135 "test/fpy/test_imports.py::TestImportDiamond::test_star_import_diamond_folds"), [3](test/fpy/test_imports.py#L2150 "test/fpy/test_imports.py::TestImportDiamond::test_member_import_diamond_folds")

> To use several of a sequence's symbols, import it whole and qualify them, use one `from seq import a, b`, or use `from seq import *`.

*Tests:* [1](test/fpy/test_imports.py#L2054 "test/fpy/test_imports.py::TestImportDuplicateSequence::test_import_and_from_same_sequence_coexist"), [2](test/fpy/test_imports.py#L2078 "test/fpy/test_imports.py::TestImportDuplicateSequence::test_two_from_same_sequence_coexist"), [3](test/fpy/test_imports.py#L888 "test/fpy/test_imports.py::TestImportDuplicates::test_import_same_sequence_twice_is_idempotent"), [4](test/fpy/test_imports.py#L2425 "test/fpy/test_imports.py::TestImportRelative::test_relative_and_absolute_same_file_idempotent")

An imported sequence may contain only function definitions and import statements. If it contains any other top-level statement, an error is raised. Such a statement is a side effect that would have to execute at the position of the import; because an imported sequence is compiled as a sibling scope block that does not run inline, it has no place to execute.

*Tests:* [1](test/fpy/test_imports.py#L167 "test/fpy/test_imports.py::TestImportSideEffects::test_side_effecting_import_is_error"), [2](test/fpy/test_imports.py#L181 "test/fpy/test_imports.py::TestImportSideEffects::test_functions_only_sequence_compiles"), [3](test/fpy/test_imports.py#L1467 "test/fpy/test_imports.py::TestImportVariables::test_top_level_variable_is_error"), [4](test/fpy/test_imports.py#L2023 "test/fpy/test_imports.py::TestImportFrom::test_from_import_side_effects_is_error"), [5](test/fpy/test_imports.py#L739 "test/fpy/test_imports.py::TestImportedSequenceShadowingBuiltins::test_imported_sequence_cannot_declare_top_level_flags"), [6](test/fpy/test_imports.py#L518 "test/fpy/test_imports.py::TestImportFileErrors::test_empty_sequence_compiles_without_warning")

If an importing sequence names a [definition](#definitions) of an imported sequence by a name that begins with an underscore, the `import-underscore` warning is emitted.

*Tests:* [1](test/fpy/test_imports.py#L216 "test/fpy/test_imports.py::TestImportUnderscore::test_underscore_module_access_warns"), [2](test/fpy/test_imports.py#L232 "test/fpy/test_imports.py::TestImportUnderscore::test_underscore_from_import_warns"), [3](test/fpy/test_imports.py#L280 "test/fpy/test_imports.py::TestImportUnderscore::test_library_internal_use_does_not_warn"), [4](test/fpy/test_imports.py#L293 "test/fpy/test_imports.py::TestImportUnderscore::test_underscore_alias_statement_warns_but_uses_do_not")

> A leading underscore marks a definition as internal to its sequence. The warning is on the importer's mention of the name -- `lib._helper()`, `import lib._helper`, `from lib import _helper` -- never on the imported sequence's own references to it. An alias renames: after `from lib import _helper as helper`, uses of `helper` do not warn, though the `from` statement itself already did.




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

`sleep(seconds: U32 = 0, useconds: U32 = 0)`

#### Semantics

At evaluation, the program [sleeps](#sleeping) for a duration of `seconds` seconds and `useconds` microseconds.

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
`U8`, `U16`, `U32`, and `U64` are the primitive unsigned integer types with bitwidths 8, 16, 32 and 64, respectively. They use the standard binary representation of unsigned integers.

`I8`, `I16`, `I32`, and `I64` are the primitive signed integer types with bitwidths 8, 16, 32 and 64, respectively. They use the standard two's complement representation of signed integers.

`F32`, and `F64` are the primitive IEEE floating-point types with bitwidths 32 and 64, respectively.

> There are other numerical types such as [Int or Float](#internal-types) which are not primitive.

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

> Modules, type names, and function names are valid expressions syntactically, but not semantically. Thus, trying to access a member of either of these symbols will raise an error.

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

Compile-time constant floats (including literals and constant-folded expressions) can only be narrowed into a smaller floating-point type when the value lies inside the destination's representable range. When the value fits, the compiler rounds it to the nearest representable floating-point number; otherwise compilation fails with an out-of-range error.


# Execution

## Control flow
A **branch** is a block of code which conditionally executes.

> That is, whether or not that code executes depends on some expression.

The list of statements which have branches is:
* The [if statement](#ifs)
* The [while loop statement](#while-loop-statement)
* The [for loop statement](#for-loop-statement)
* The [check statement](#check-statement)
TODO how does this definition handle assert/exit?

## Sleeping
The program may **sleep** until an absolute time called the **wakeup time**.

While the current time is before the wakeup time, no statements may execute

The program may also sleep for a time duration. This is the same as sleeping with a wakeup time of now, plus the specified duration.
