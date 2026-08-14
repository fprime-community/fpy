from typing import NamedTuple


class AstDotExpr(NamedTuple):
    # this is ALWAYS member access
    parent: AstExpr
    ident: AstIdent


type AstName = str


class AstProperQualifiedName(NamedTuple):
    qualifier: "AstQualifiedName"
    name: AstName


type AstQualifiedName = AstProperQualifiedName | AstName
