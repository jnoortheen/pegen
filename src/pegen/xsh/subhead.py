import ast
import enum
import io
import os
import sys
import tokenize
from typing import (
    Any, List, Tuple, TypeVar, Union, NoReturn, Optional, Callable, Literal, Iterator
)

from pegen.parser import Parser
from pegen.tokenizer import Tokenizer

# Singleton ast nodes, created once for efficiency
Load = ast.Load()
Store = ast.Store()
Del = ast.Del()

Node = TypeVar("Node")
FC = TypeVar("FC", ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

EXPR_NAME_MAPPING = {
    ast.Attribute: "attribute",
    ast.Subscript: "subscript",
    ast.Starred: "starred",
    ast.Name: "name",
    ast.List: "list",
    ast.Tuple: "tuple",
    ast.Lambda: "lambda",
    ast.Call: "function call",
    ast.BoolOp: "expression",
    ast.BinOp: "expression",
    ast.UnaryOp: "expression",
    ast.GeneratorExp: "generator expression",
    ast.Yield: "yield expression",
    ast.YieldFrom: "yield expression",
    ast.Await: "await expression",
    ast.ListComp: "list comprehension",
    ast.SetComp: "set comprehension",
    ast.DictComp: "dict comprehension",
    ast.Dict: "dict literal",
    ast.Set: "set display",
    ast.JoinedStr: "f-string expression",
    ast.FormattedValue: "f-string expression",
    ast.Compare: "comparison",
    ast.IfExp: "conditional expression",
    ast.NamedExpr: "named expression",
}


class Target(enum.Enum):
    FOR_TARGETS = enum.auto()
    STAR_TARGETS = enum.auto()
    DEL_TARGETS = enum.auto()



class ParserBase(Parser):

    #: Name of the source file, used in error reports
    filename : str

    def __init__(self,
        tokenizer: Tokenizer, *,
        verbose: bool = False,
        filename: str = "<unknown>",
        py_version: Optional[tuple] = None,
    ) -> None:
        super().__init__(tokenizer, verbose=verbose)
        self.filename = filename
        self.py_version = min(py_version, sys.version_info) if py_version else sys.version_info

    def parse(self, rule: str, call_invalid_rules: bool = False) -> Optional[ast.AST]:
        old = self.call_invalid_rules
        self.call_invalid_rules = call_invalid_rules
        res = getattr(self, rule)()

        if res is None:

            # Grab the last token that was parsed in the first run to avoid
            # polluting a generic error reports with progress made by invalid rules.
            last_token = self._tokenizer.diagnose()

            if not call_invalid_rules:
                self.call_invalid_rules = True

                # Reset the parser cache to be able to restart parsing from the
                # beginning.
                self._reset(0)  # type: ignore
                self._cache.clear()

                res = getattr(self, rule)()

            self.raise_raw_syntax_error("invalid syntax", last_token.start, last_token.end)

        return res

    @classmethod
    def parse_file(
            cls,
            path: str,
            py_version: Optional[tuple] = None,
            token_stream_factory: Optional[
                Callable[[Callable[[], str]], Iterator[tokenize.TokenInfo]]
            ] = None,
            verbose: bool = False,
    ) -> ast.Module:
        """Parse a file."""
        with open(path) as f:
            tok_stream = (
                token_stream_factory(f.readline)
                if token_stream_factory else
                tokenize.generate_tokens(f.readline)
            )
            tokenizer = Tokenizer(tok_stream, verbose=verbose, path=path)
            parser = cls(
                tokenizer,
                verbose=verbose,
                filename=os.path.basename(path),
                py_version=py_version
            )
            return parser.parse("file")

    @classmethod
    def parse_string(
            cls,
            source: str,
            mode: Union[Literal["eval"], Literal["exec"]],
            py_version: Optional[tuple] = None,
            token_stream_factory: Optional[
                Callable[[Callable[[], str]], Iterator[tokenize.TokenInfo]]
            ] = None,
            verbose: bool = False,
    ) -> Any:
        """Parse a string."""
        tok_stream = (
            token_stream_factory(io.StringIO(source).readline)
            if token_stream_factory else
            tokenize.generate_tokens(io.StringIO(source).readline)
        )
        tokenizer = Tokenizer(tok_stream, verbose=verbose)
        parser = cls(tokenizer, verbose=verbose, py_version=py_version)
        return parser.parse(mode if mode == "eval" else "file")

    def check_version(self, min_version: Tuple[int, ...], error_msg: str, node: Node) -> Node:
        """Check that the python version is high enough for a rule to apply.

        """
        if self.py_version >= min_version:
            return node
        else:
            raise SyntaxError(
                f"{error_msg} is only supported in Python {min_version} and above."
            )

    def raise_indentation_error(self, msg: str) -> None:
        """Raise an indentation error."""
        last_token = self._tokenizer.diagnose()
        args = (self.filename, last_token.start[0], last_token.start[1] + 1, last_token.line)
        if sys.version_info >= (3, 10):
            args += (last_token.end[0], last_token.end[1] + 1)
        raise IndentationError(msg, args)

    def get_expr_name(self, node) -> str:
        """Get a descriptive name for an expression."""
        # See https://github.com/python/cpython/blob/master/Parser/pegen.c#L161
        assert node is not None
        node_t = type(node)
        if node_t is ast.Constant:
            v = node.value
            if v is Ellipsis:
                return "ellipsis"
            elif v is None:
                return str(v)
            # Avoid treating 1 as True through == comparison
            elif v is True:
                return str(v)
            elif v is False:
                return str(v)
            else:
                return "literal"

        try:
            return EXPR_NAME_MAPPING[node_t]
        except KeyError:
            raise ValueError(
                f"unexpected expression in assignment {type(node).__name__} "
                f"(line {node.lineno})."
            )

    def get_invalid_target(self, target: Target, node: Optional[ast.AST]) -> Optional[ast.AST]:
        """Get the meaningful invalid target for different assignment type."""
        if node is None:
            return None

        # We only need to visit List and Tuple nodes recursively as those
        # are the only ones that can contain valid names in targets when
        # they are parsed as expressions. Any other kind of expression
        # that is a container (like Sets or Dicts) is directly invalid and
        # we do not need to visit it recursively.
        if isinstance(node, (ast.List, ast.Tuple)):
            for e in node.elts:
                if (inv := self.get_invalid_target(target, e)) is not None:
                    return inv
        elif isinstance(node, ast.Starred):
            if target is Target.DEL_TARGETS:
                return node
            return self.get_invalid_target(target, node.value)
        elif isinstance(node, ast.Compare):
            # This is needed, because the `a in b` in `for a in b` gets parsed
            # as a comparison, and so we need to search the left side of the comparison
            # for invalid targets.
            if target is Target.FOR_TARGETS:
                if isinstance(node.ops[0], ast.In):
                    return self.get_invalid_target(target, node.left)
                return None

            return node
        elif isinstance(node, (ast.Name, ast.Subscript, ast.Attribute)):
            return None
        else:
            return node

    def set_expr_context(self, node, context):
        """Set the context (Load, Store, Del) of an ast node."""
        node.ctx = context
        return node

    def ensure_real(self, number: ast.Constant):
        value = ast.literal_eval(number.string)
        if type(value) is complex:
            self.raise_syntax_error_known_location("real number required in complex literal", number)
        return value

    def ensure_imaginary(self, number:  ast.Constant):
        value = ast.literal_eval(number.string)
        if type(value) is not complex:
            self.raise_syntax_error_known_location("imaginary number required in complex literal", number)
        return value

    def generate_ast_for_string(self, tokens):
        """Generate AST nodes for strings."""
        err_args = None
        line_offset = tokens[0].start[0]
        line = line_offset
        col_offset = 0
        source = "(\n"
        for t in tokens:
            n_line = t.start[0] - line
            if n_line:
                col_offset = 0
            source += "\n" * n_line + ' ' * (t.start[1] - col_offset) + t.string
            line, col_offset = t.end
        source += "\n)"
        try:
            m = ast.parse(source)
        except SyntaxError as err:
            args = (err.filename, err.lineno + line_offset - 2, err.offset, err.text)
            if sys.version_info >= (3, 10):
                args += (err.end_lineno + line_offset - 2, err.end_offset)
            err_args = (err.msg, args)
            # Ensure we do not keep the frame alive longer than necessary
            # by explicitely deleting the error once we got what we needed out
            # of it
            del err

        # Avoid getting a triple nesting in the error report that does not
        # bring anything relevant to the traceback.
        if err_args is not None:
            raise SyntaxError(*err_args)

        node = m.body[0].value
        # Since we asked Python to parse an alterred source starting at line 2
        # we alter the lineno of the returned AST to recover the right line.
        # If the string start at line 1, tha AST says 2 so we need to decrement by 1
        # hence the -2.
        ast.increment_lineno(node, line_offset - 2)
        return node

    def extract_import_level(self, tokens: List[tokenize.TokenInfo]) -> int:
        """Extract the relative import level from the tokens preceding the module name.

        '.' count for one and '...' for 3.

        """
        level = 0
        for t in tokens:
            if t.string == ".":
                level += 1
            else:
                level += 3
        return level

    def set_decorators(self,
        target: FC,
        decorators: list
    ) -> FC:
        """Set the decorators on a function or class definition."""
        target.decorator_list = decorators
        return target

    def get_comparison_ops(self, pairs):
        return [op for op, _ in pairs]

    def get_comparators(self, pairs):
        return [comp for _, comp in pairs]

    def set_arg_type_comment(self, arg, type_comment):
        if type_comment or sys.version_info < (3, 9):
            arg.type_comment = type_comment
        return arg

    def make_arguments(self,
        pos_only: Optional[List[Tuple[ast.arg, None]]],
        pos_only_with_default: List[Tuple[ast.arg, Any]],
        param_no_default: Optional[List[Tuple[ast.arg, None]]],
        param_default: Optional[List[Tuple[ast.arg, Any]]],
        after_star: Optional[Tuple[Optional[ast.arg], List[Tuple[ast.arg, Any]], Optional[ast.arg]]]
    ) -> ast.arguments:
        """Build a function definition arguments."""
        defaults = (
            [d for _, d in pos_only_with_default if d is not None]
            if pos_only_with_default else
            []
        )
        defaults += (
            [d for _, d in param_default if d is not None]
            if param_default else
            []
        )

        pos_only = pos_only or pos_only_with_default

        # Because we need to combine pos only with and without default even
        # the version with no default is a tuple
        pos_only = [p for p, _ in pos_only]
        params = (param_no_default or []) + ([p for p, _ in param_default] if param_default else [])

        # If after_star is None, make a default tuple
        after_star = after_star or (None, [], None)

        return ast.arguments(
            posonlyargs=pos_only,
            args=params,
            defaults=defaults,
            vararg=after_star[0],
            kwonlyargs=[p for p, _ in after_star[1]],
            kw_defaults=[d for _, d in after_star[1]],
            kwarg=after_star[2]
        )

    def _build_syntax_error(
        self,
        message: str,
        start: Optional[Tuple[int, int]] = None,
        end: Optional[Tuple[int, int]] = None
    ) -> None:
        line_from_token = start is None and end is None
        if start is None or end is None:
            tok = self._tokenizer.diagnose()
            start = start or tok.start
            end = end or tok.end

        if line_from_token:
            line = tok.line
        else:
            # End is used only to get the proper text
            line = "\\n".join(
                self._tokenizer.get_lines(list(range(start[0], end[0] + 1)))
            )

        # tokenize.py index column offset from 0 while Cpython index column
        # offset at 1 when reporting SyntaxError, so we need to increment
        # the column offset when reporting the error.
        args = (self.filename, start[0], start[1] + 1, line)
        if sys.version_info >= (3, 10):
            args += (end[0], end[1] + 1)

        return SyntaxError(message, args)

    def raise_raw_syntax_error(
        self,
        message: str,
        start: Optional[Tuple[int, int]] = None,
        end: Optional[Tuple[int, int]] = None
    ) -> NoReturn:
        raise self._build_syntax_error(message, start, end)

    def make_syntax_error(self, message: str) -> None:
        return self._build_syntax_error(message)

    def expect_forced(self, res: Any, expectation: str) -> Optional[tokenize.TokenInfo]:
        if res is None:
            last_token = self._tokenizer.diagnose()
            self.raise_raw_syntax_error(
                f"expected {expectation}", last_token.start, last_token.start
            )
        return res

    def raise_syntax_error(self, message: str) -> NoReturn:
        """Raise a syntax error."""
        tok = self._tokenizer.diagnose()
        raise self._build_syntax_error(message, tok.start, tok.end if tok.type != 4 else tok.start)

    def raise_syntax_error_known_location(
        self, message: str, node: Union[ast.AST, tokenize.TokenInfo]
    ) -> NoReturn:
        """Raise a syntax error that occured at a given AST node."""
        if isinstance(node, tokenize.TokenInfo):
            start = node.start
            end = node.end
        else:
            start = node.lineno, node.col_offset
            end = node.end_lineno, node.end_col_offset

        raise self._build_syntax_error(message, start, end)

    def raise_syntax_error_known_range(
        self,
        message: str,
        start_node: Union[ast.AST, tokenize.TokenInfo],
        end_node: Union[ast.AST, tokenize.TokenInfo]
    ) -> NoReturn:
        if isinstance(start_node, tokenize.TokenInfo):
            start = start_node.start
        else:
            start = start_node.lineno, start_node.col_offset

        if isinstance(end_node, tokenize.TokenInfo):
            end = end_node.end
        else:
            end = end_node.end_lineno, end_node.end_col_offset

        raise self._build_syntax_error(message, start, end)

    def raise_syntax_error_starting_from(
        self,
        message: str,
        start_node: Union[ast.AST, tokenize.TokenInfo]
    ) -> NoReturn:
        if isinstance(start_node, tokenize.TokenInfo):
            start = start_node.start
        else:
            start = start_node.lineno, start_node.col_offset

        last_token = self._tokenizer.diagnose()

        raise self._build_syntax_error(message, start, last_token.start)

    def raise_syntax_error_invalid_target(
        self, target: Target, node: Optional[ast.AST]
    ) -> NoReturn:
        invalid_target = self.get_invalid_target(target, node)

        if invalid_target is None:
            return None

        if target in (Target.STAR_TARGETS, Target.FOR_TARGETS):
            msg = f"cannot assign to {self.get_expr_name(invalid_target)}"
        else:
            msg = f"cannot delete {self.get_expr_name(invalid_target)}"

        self.raise_syntax_error_known_location(msg, invalid_target)

