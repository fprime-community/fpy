from typing import NamedTuple

# stage 2 is
# * explicit dotexpr vs qual name split
# * no check stmts
# * no anon exprs?
# * default values that were FpyValues become syntax? or maybe we have an AstConst which can have an FpyValue. Actually maybe we should just have this anyways
# * all callable args are positional, default values are explicit


class AstDotExpr(NamedTuple):
    # this is ALWAYS member access
    parent: AstExpr
    ident: AstIdent


type AstName = str


class AstProperQualifiedName(NamedTuple):
    qualifier: "AstQualifiedName"
    name: AstName


type AstQualifiedName = AstProperQualifiedName | AstName


class AstAssign(NamedTuple):
    assignee: "AstAssignee"
    value: AstExpr


class AstDefineVar(NamedTuple):
    var: AstName
    type: AstQualifiedName
    initial_value: AstExpr


class AstAssigneeMember(NamedTuple):
    parent: "AstAssignee"
    # why shouldn't it be a str? because it's not a name. it doesn't refer to any symbol
    member: str


class AstAssigneeElement(NamedTuple):
    parent: "AstAssignee"
    element: AstExpr


# something we can assign to
type AstAssignee = AstName | AstAssigneeMember | AstAssigneeElement
