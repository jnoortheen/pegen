""""Conftest for pure python parser."""
from pathlib import Path

import pytest
from pegen.build import build_parser
from pegen.python_generator import PythonParserGenerator
from pegen.utils import generate_parser, import_file
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
