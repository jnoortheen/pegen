""""Conftest for pure python parser."""
from pathlib import Path

import pytest
from pegen.build import build_parser
from pegen.python_generator import PythonParserGenerator
from pegen.utils import import_file
from .tools import nodes_equal

import io


@pytest.fixture(scope="session")
def parser():
    grammar_path = Path(__file__).parent.parent.parent / "src/xsh_parser/xonsh.peg"
    grammar = build_parser(grammar_path)[0]

    out = io.StringIO()
    genr = PythonParserGenerator(grammar, out)
    genr.generate("<string>")
    parser_path = str(Path(__file__).parent / "parser_cache" / "parser.py")
    with open(parser_path, "w") as f:
        f.write(out.getvalue())
    return import_file("xonsh_parser", parser_path)


@pytest.fixture(scope="session")
def python_parser_cls(parser):
    return getattr(parser, "XonshParser")


@pytest.fixture(scope="session")
def python_parse_file(parser):
    return getattr(parser, "parse_file")


@pytest.fixture(scope="session")
def python_parse_str(parser):
    return getattr(parser, "parse_string")


@pytest.fixture()
def ctx_parse():
    """contextual parse"""

    def parse(input: str, **ctx):
        tree = execer.parse(input, ctx=set(ctx.keys()))
        return tree

    return parse


@pytest.fixture
def check_ast(parser):
    import ast

    def factory(inp: str, run=True, mode="eval", globals=None, locals=None):
        # __tracebackhide__ = True
        # expect a Python AST
        exp = ast.parse(inp, mode=mode)
        # observe something from xonsh
        obs = parser.parse_string(inp, mode=mode)
        # Check that they are equal
        assert nodes_equal(exp, obs)
        # round trip by running xonsh AST via Python
        if run:
            exec(compile(obs, "<test-ast>", mode), globals, locals)

    return factory


@pytest.fixture
def check_stmts(check_ast):
    def factory(inp, run=True, mode="exec"):
        __tracebackhide__ = True
        if not inp.endswith("\n"):
            inp += "\n"
        check_ast(inp, run=run, mode=mode)

    return factory


@pytest.fixture
def check_xonsh_ast(parser):
    def factory(
        inp: str,
        run=True,
        mode="eval",
        return_obs=False,
        globals=None,
        locals=None,
    ):
        obs = parser.parse_string(inp, mode="exec")
        if obs is None:
            return  # comment only
        bytecode = compile(obs, "<test-xonsh-ast>", mode)
        if run:
            exec(bytecode, globals, locals)
        return obs if return_obs else True

    return factory


@pytest.fixture
def unparse(parser):
    def factory(inp: str):
        import ast

        tree = parser.parse_string(inp, mode="exec")
        if tree is None:
            return  # comment only
        return ast.unparse(tree)

    return factory


@pytest.fixture
def check_xonsh(check_xonsh_ast):
    def factory(xenv, inp, run=True, mode="exec"):
        __tracebackhide__ = True
        if not inp.endswith("\n"):
            inp += "\n"
        check_xonsh_ast(xenv, inp, run=run, mode=mode)

    return factory


@pytest.fixture
def eval_code(parser):
    def factory(inp, mode="eval", **loc_vars):
        obs = parser.parse_string(inp, mode=mode)
        bytecode = compile(obs, "<test-xonsh-ast>", mode)
        return eval(bytecode, loc_vars)

    return factory
