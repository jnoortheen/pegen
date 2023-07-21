"""Tests the xonsh parser."""
import itertools
from pathlib import Path

import pytest


data_dir = Path(__file__).parent / "data"


def _get_check_ast_params():
    for idx, line in enumerate((data_dir / "check_ast.txt").read_text().splitlines()):
        if "# " not in line:
            continue
        expr, name = line.rsplit("# ", 1)
        yield expr.strip(), f"{idx+1}_{name}"


@pytest.mark.parametrize(
    "expr", [pytest.param(expr, id=name) for expr, name in _get_check_ast_params()]
)
def test_check_ast(check_ast, expr):
    check_ast(expr, run=False)


def test_subscription_syntaxes(eval_code):
    assert eval_code("[1, 2, 3][-1]") == 3
    assert eval_code("[1, 2, 3][-1]") == 3
    assert eval_code("'string'[-1]") == "g"


def test_subscription_special_syntaxes(eval_code):
    class Arr:
        def __getitem__(self, item):
            return item

    arr_container = Arr()
    d = {}

    assert eval_code("arr[1, 2, 3]", arr=arr_container) == (1, 2, 3)
    # dataframe
    assert eval_code('arr[["a", "b"]]', arr=arr_container) == ["a", "b"]
    eval_code("d[arr.__name__]=True", arr=arr_container, d=d)
    assert d == {"Arr": True}
    # extslice
    assert eval_code('arr[:, "2"]') == 2


@pytest.mark.parametrize("third", [True, False])
@pytest.mark.parametrize("second", [True, False])
@pytest.mark.parametrize("first", [True, False])
def test_dict_from_dict_three_xyz(first, second, third, check_ast):
    val1 = '"x": 1' if first else '**{"x": 1}'
    val2 = '"y": 2' if second else '**{"y": 2}'
    val3 = '"z": 3' if third else '**{"z": 3}'
    check_ast("{" + val1 + "," + val2 + "," + val3 + "}")


#
# statements
#


def test_matmult_eq(check_stmts):
    check_stmts("x @= y", False)


@pytest.mark.parametrize(
    "inp",
    [
        pytest.param("x = 42", id="equals"),
        pytest.param("x = 42;", id="equals_semi"),
        pytest.param("x = y = 42", id="x_y_equals"),
        pytest.param("x = y = 42;", id="x_y_equals_semi"),
        pytest.param("x = 42; y = 65", id="equals_two"),
        pytest.param("x = 42; y = 65;", id="equals_two_semi"),
        pytest.param("x = 42; y = 65; z = 6", id="equals_three"),
        pytest.param("x = 42; y = 65; z = 6;", id="equals_three_semi"),
        pytest.param("x = 42; x += 65", id="plus_eq"),
        pytest.param("x = 42; x -= 2", id="sub_eq"),
        pytest.param("x = 42; x *= 2", id="times_eq"),
        pytest.param("x = 42; x /= 2", id="div_eq"),
        pytest.param("x = 42; x //= 2", id="floordiv_eq"),
        pytest.param("x = 42; x **= 2", id="pow_eq"),
        pytest.param("x = 42; x %= 2", id="mod_eq"),
        pytest.param("x = 42; x ^= 2", id="xor_eq"),
        pytest.param("x = 42; x &= 2", id="ampersand_eq"),
        pytest.param("x = 42; x |= 2", id="bitor_eq"),
        pytest.param("x = 42; x <<= 2", id="lshift_eq"),
        pytest.param("x = 42; x >>= 2", id="rshift_eq"),
        pytest.param("x, y = 42, 65", id="bare_unpack"),
        pytest.param("(x, y) = 42, 65", id="lhand_group_unpack"),
        pytest.param("x, y = (42, 65)", id="rhand_group_unpack"),
        pytest.param("(x, y) = (42, 65)", id="grouped_unpack"),
        pytest.param("(x, y) = (z, a) = (7, 8)", id="double_grouped_unpack"),
        pytest.param("x, y = z, a = 7, 8", id="double_ungrouped_unpack"),
        pytest.param("*y, = [1, 2, 3]", id="stary_eq"),
        pytest.param("*y, x = [1, 2, 3]", id="stary_x"),
        pytest.param("(x, *y) = [1, 2, 3]", id="tuple_x_stary"),
        pytest.param("[x, *y] = [1, 2, 3]", id="list_x_stary"),
        pytest.param("x, *y = [1, 2, 3]", id="bare_x_stary"),
        pytest.param("x, *y, z = [1, 2, 2, 3]", id="bare_x_stary_z"),
        pytest.param("x = [42]; x[0] = 65", id="equals_list"),
        pytest.param("x = {42: 65}; x[42] = 3", id="equals_dict"),
        pytest.param("class X(object):\n  pass\nx = X()\nx.a = 65", id="equals_attr"),
        pytest.param("x : int = 42", id="equals_annotation"),
        pytest.param("x : int", id="equals_annotation_empty"),
        pytest.param('x = {"x": 1}\nx.keys()', id="dict_keys"),
        pytest.param('assert True, "wow mom"', id="assert_msg"),
        pytest.param("assert True", id="assert"),
        pytest.param("pass", id="pass"),
        pytest.param("x = 42; del x", id="del"),
        pytest.param("x = 42; del x,", id="del_comma"),
        pytest.param("x = 42; y = 65; del x, y", id="del_two"),
        pytest.param("x = 42; y = 65; del x, y,", id="del_two_comma"),
        pytest.param("x = 42; y = 65; del (x, y)", id="del_with_parens"),
        pytest.param("if True:\n  pass", id="if_true"),
        pytest.param("if True:\n  pass\n  pass", id="if_true_twolines"),
        pytest.param("if True:\n  pass\n  pass\npass", id="if_true_twolines_deindent"),
        pytest.param("if True:\n  pass\nelse: \n  pass", id="if_true_else"),
        pytest.param("if True:\n  x = 42", id="if_true_x"),
        pytest.param("x = 42\nif x == 1:\n  pass", id="if_switch"),
        pytest.param(
            "x = 42\nif x == 1:\n  pass\n" "elif x == 2:\n  pass\nelse:\n  pass",
            id="if_switch_elif1_else",
        ),
        pytest.param(
            "x = 42\nif x == 1:\n  pass\n"
            "elif x == 2:\n  pass\n"
            "elif x == 3:\n  pass\n"
            "elif x == 4:\n  pass\n"
            "else:\n  pass",
            id="if_switch_elif2_else",
        ),
        pytest.param(
            "x = 42\nif x == 1:\n  pass\n  if x == 4:\n     pass", id="if_nested"
        ),
        pytest.param("while False:\n  pass", id="while_false"),
        pytest.param("while False:\n  pass\nelse:\n  pass", id="while_false_else"),
        pytest.param("for x in range(6):\n  pass", id="for"),
        pytest.param('for x, y in zip(range(6), "123456"):\n  pass', id="for_zip"),
        pytest.param("x = [42]\nfor x[0] in range(3):\n  pass", id="for_idx"),
        pytest.param(
            'x = [42]\nfor x[0], y in zip(range(6), "123456"):\n' "  pass",
            id="for_zip_idx",
        ),
        pytest.param("def f():\n  pass", id="func"),
        pytest.param("def f():\n  return", id="func_ret"),
        pytest.param("def f():\n  return 42", id="func_ret_42"),
        pytest.param("def f():\n  return 42, 65", id="func_ret_42_65"),
        pytest.param("def f() -> int:\n  pass", id="func_rarrow"),
        pytest.param("def f(x):\n  return x", id="func_x"),
        pytest.param("def f(x=42):\n  return x", id="func_kwx"),
        pytest.param("def f(x, y):\n  return x", id="func_x_y"),
        pytest.param("def f(x, y, z):\n  return x", id="func_x_y_z"),
        pytest.param("def f(x, y=42):\n  return x", id="func_x_kwy"),
        pytest.param("def f(x=65, y=42):\n  return x", id="func_kwx_kwy"),
        pytest.param("def f(x=65, y=42, z=1):\n  return x", id="func_kwx_kwy_kwz"),
        pytest.param("def f(x,):\n  return x", id="func_x_comma"),
        pytest.param("def f(x, y,):\n  return x", id="func_x_y_comma"),
        pytest.param("def f(x, y, z,):\n  return x", id="func_x_y_z_comma"),
        pytest.param("def f(x, y=42,):\n  return x", id="func_x_kwy_comma"),
        pytest.param("def f(x=65, y=42,):\n  return x", id="func_kwx_kwy_comma"),
        pytest.param(
            "def f(x=65, y=42, z=1,):\n  return x", id="func_kwx_kwy_kwz_comma"
        ),
        pytest.param("def f(*args):\n  return 42", id="func_args"),
        pytest.param("def f(*args, x):\n  return 42", id="func_args_x"),
        pytest.param("def f(*args, x, y):\n  return 42", id="func_args_x_y"),
        pytest.param("def f(*args, x, y=10):\n  return 42", id="func_args_x_kwy"),
        pytest.param("def f(*args, x=10, y):\n  return 42", id="func_args_kwx_y"),
        pytest.param("def f(*args, x=42, y=65):\n  return 42", id="func_args_kwx_kwy"),
        pytest.param("def f(x, *args):\n  return 42", id="func_x_args"),
        pytest.param("def f(x, *args, y):\n  return 42", id="func_x_args_y"),
        pytest.param("def f(x, *args, y, z):\n  return 42", id="func_x_args_y_z"),
        pytest.param("def f(**kwargs):\n  return 42", id="func_kwargs"),
        pytest.param("def f(x, **kwargs):\n  return 42", id="func_x_kwargs"),
        pytest.param("def f(x, y, **kwargs):\n  return 42", id="func_x_y_kwargs"),
        pytest.param("def f(x, y=42, **kwargs):\n  return 42", id="func_x_kwy_kwargs"),
        pytest.param("def f(*args, **kwargs):\n  return 42", id="func_args_kwargs"),
        pytest.param(
            "def f(x, *args, **kwargs):\n  return 42", id="func_x_args_kwargs"
        ),
        pytest.param(
            "def f(x, y, *args, **kwargs):\n  return 42", id="func_x_y_args_kwargs"
        ),
        pytest.param(
            "def f(x=10, *args, **kwargs):\n  return 42", id="func_kwx_args_kwargs"
        ),
        pytest.param(
            "def f(x, y=42, *args, **kwargs):\n  return 42", id="func_x_kwy_args_kwargs"
        ),
        pytest.param(
            "def f(x, *args, y, **kwargs):\n  return 42", id="func_x_args_y_kwargs"
        ),
        pytest.param(
            "def f(x, *args, y=42, **kwargs):\n  return 42", id="func_x_args_kwy_kwargs"
        ),
        pytest.param(
            "def f(*args, y, **kwargs):\n  return 42", id="func_args_y_kwargs"
        ),
        pytest.param("def f(*, x):\n  return 42", id="func_star_x"),
        pytest.param("def f(*, x, y):\n  return 42", id="func_star_x_y"),
        pytest.param("def f(*, x, **kwargs):\n  return 42", id="func_star_x_kwargs"),
        pytest.param(
            "def f(*, x=42, **kwargs):\n  return 42", id="func_star_kwx_kwargs"
        ),
        pytest.param("def f(x, *, y):\n  return 42", id="func_x_star_y"),
        pytest.param("def f(x, y, *, z):\n  return 42", id="func_x_y_star_z"),
        pytest.param("def f(x, y=42, *, z):\n  return 42", id="func_x_kwy_star_y"),
        pytest.param("def f(x, y=42, *, z=65):\n  return 42", id="func_x_kwy_star_kwy"),
        pytest.param(
            "def f(x, *, y, **kwargs):\n  return 42", id="func_x_star_y_kwargs"
        ),
        pytest.param("def f(x:int):\n  return x", id="func_tx"),
        pytest.param("def f(x:int, y:float=10.0):\n  return x", id="func_txy"),
    ],
)
def test_statements(check_stmts, inp):
    check_stmts(inp)


@pytest.mark.parametrize(
    "inp",
    [
        pytest.param("raise", id="raise"),
        pytest.param("raise TypeError", id="raise_x"),
        pytest.param("raise TypeError from x", id="raise_x_from"),
        pytest.param("import x", id="import_x"),
        pytest.param("import x.y", id="import_xy"),
        pytest.param("import x.y.z", id="import_xyz"),
        pytest.param("from x import y", id="from_x_import_y"),
        pytest.param("from . import y", id="from_dot_import_y"),
        pytest.param("from .x import y", id="from_dotx_import_y"),
        pytest.param("from ..x import y", id="from_dotdotx_import_y"),
        pytest.param("from ...x import y", id="from_dotdotdotx_import_y"),
        pytest.param("from ....x import y", id="from_dotdotdotdotx_import_y"),
        pytest.param("import x, y", id="import_x_y"),
        pytest.param("import x, y, z", id="import_x_y_z"),
        pytest.param("from x import y, z", id="from_x_import_y_z"),
        pytest.param("from x import y as z", id="from_x_import_y_as_z"),
        pytest.param("from x import y as z, a as b", id="from_x_import_y_as_z_a_as_b"),
        pytest.param("from x import (y, z)", id="from_x_import_group_y_z"),
        pytest.param("from x import (y, z,)", id="from_x_import_group_y_z_comma"),
        pytest.param("from x import (y, z as a)", id="from_x_import_group_y_z_as_a"),
        pytest.param(
            "from x import (y, z as a,)", id="from_x_import_group_y_z_as_a_comma"
        ),
        pytest.param(
            "from x import (y, z as a, b as c)",
            id="from_x_import_group_y_z_as_a_b_as_c",
        ),
        pytest.param("from x import *", id="from_x_import_star"),
        pytest.param("from x import (x, y, z)", id="from_x_import_group_x_y_z"),
        pytest.param("from x import (x, y, z,)", id="from_x_import_group_x_y_z_comma"),
        pytest.param("from x import y as z", id="from_x_import_y_as_z"),
        pytest.param("from x import y as z, a as b", id="from_x_import_y_as_z_a_as_b"),
        pytest.param("from .x import y as z, a as b, c as d", id="from_dotx_import_y"),
        pytest.param("continue", id="continue"),
        pytest.param("break", id="break"),
        pytest.param("global x", id="global_x"),
        pytest.param("global x, y", id="global_xy"),
        pytest.param("nonlocal x", id="nonlocal_x"),
        pytest.param("nonlocal x, y", id="nonlocal_xy"),
        pytest.param("yield", id="yield"),
        pytest.param("yield x", id="yield_x"),
        pytest.param("yield x,", id="yield_x_comma"),
        pytest.param("yield x, y", id="yield_x_y"),
        pytest.param("yield from x", id="yield_from_x"),
        pytest.param("return", id="return"),
        pytest.param("return x", id="return_x"),
        pytest.param("return x,", id="return_x_comma"),
        pytest.param("return x, y", id="return_x_y"),
    ],
)
def test_statements_no_run(check_stmts, inp):
    check_stmts(inp, run=False)


def test_for_attr(check_stmts):
    check_stmts("for x.a in range(3):\n  pass", False)


def test_for_zip_attr(check_stmts):
    check_stmts('for x.a, y in zip(range(6), "123456"):\n  pass', False)


def test_for_else(check_stmts):
    check_stmts("for x in range(6):\n  pass\nelse:  pass")


def test_async_for(check_stmts):
    check_stmts("async def f():\n    async for x in y:\n        pass\n", False)


def test_with(check_stmts):
    check_stmts("with x:\n  pass", False)


def test_with_as(check_stmts):
    check_stmts("with x as y:\n  pass", False)


def test_with_xy(check_stmts):
    check_stmts("with x, y:\n  pass", False)


def test_with_x_as_y_z(check_stmts):
    check_stmts("with x as y, z:\n  pass", False)


def test_with_x_as_y_a_as_b(check_stmts):
    check_stmts("with x as y, a as b:\n  pass", False)


def test_with_in_func(check_stmts):
    check_stmts("def f():\n    with x:\n        pass\n")


def test_async_with(check_stmts):
    check_stmts("async def f():\n    async with x as y:\n        pass\n", False)


def test_try(check_stmts):
    check_stmts("try:\n  pass\nexcept:\n  pass", False)


def test_try_except_t(check_stmts):
    check_stmts("try:\n  pass\nexcept TypeError:\n  pass", False)


def test_try_except_t_as_e(check_stmts):
    check_stmts("try:\n  pass\nexcept TypeError as e:\n  pass", False)


def test_try_except_t_u(check_stmts):
    check_stmts("try:\n  pass\nexcept (TypeError, SyntaxError):\n  pass", False)


def test_try_except_t_u_as_e(check_stmts):
    check_stmts("try:\n  pass\nexcept (TypeError, SyntaxError) as e:\n  pass", False)


def test_try_except_t_except_u(check_stmts):
    check_stmts(
        "try:\n  pass\nexcept TypeError:\n  pass\n" "except SyntaxError as f:\n  pass",
        False,
    )


def test_try_except_else(check_stmts):
    check_stmts("try:\n  pass\nexcept:\n  pass\nelse:  pass", False)


def test_try_except_finally(check_stmts):
    check_stmts("try:\n  pass\nexcept:\n  pass\nfinally:  pass", False)


def test_try_except_else_finally(check_stmts):
    check_stmts(
        "try:\n  pass\nexcept:\n  pass\nelse:\n  pass" "\nfinally:  pass", False
    )


def test_try_finally(check_stmts):
    check_stmts("try:\n  pass\nfinally:  pass", False)


def test_class(check_stmts):
    check_stmts("class X:\n  pass")


def test_class_obj(check_stmts):
    check_stmts("class X(object):\n  pass")


def test_class_int_flt(check_stmts):
    check_stmts("class X(int, object):\n  pass")


def test_class_obj_kw(check_stmts):
    # technically valid syntax, though it will fail to compile
    check_stmts("class X(object=5):\n  pass", False)


def test_decorator(check_stmts):
    check_stmts("@g\ndef f():\n  pass", False)


def test_decorator_2(check_stmts):
    check_stmts("@h\n@g\ndef f():\n  pass", False)


def test_decorator_call(check_stmts):
    check_stmts("@g()\ndef f():\n  pass", False)


def test_decorator_call_args(check_stmts):
    check_stmts("@g(x, y=10)\ndef f():\n  pass", False)


def test_decorator_dot_call_args(check_stmts):
    check_stmts("@h.g(x, y=10)\ndef f():\n  pass", False)


def test_decorator_dot_dot_call_args(check_stmts):
    check_stmts("@i.h.g(x, y=10)\ndef f():\n  pass", False)


def test_broken_prompt_func(check_stmts):
    code = "def prompt():\n" "    return '{user}'.format(\n" "       user='me')\n"
    check_stmts(code, False)


def test_class_with_methods(check_stmts):
    code = (
        "class Test:\n"
        "   def __init__(self):\n"
        '       self.msg("hello world")\n'
        "   def msg(self, m):\n"
        "      print(m)\n"
    )
    check_stmts(code, False)


def test_nested_functions(check_stmts):
    code = (
        "def test(x):\n"
        "    def test2(y):\n"
        "        return y+x\n"
        "    return test2\n"
    )
    check_stmts(code, False)


def test_function_blank_line(check_stmts):
    code = (
        "def foo():\n"
        "    ascii_art = [\n"
        '        "(╯°□°）╯︵ ┻━┻",\n'
        r'        "¯\\_(ツ)_/¯",'
        "\n"
        r'        "┻━┻︵ \\(°□°)/ ︵ ┻━┻",'
        "\n"
        "    ]\n"
        "\n"
        "    import random\n"
        "    i = random.randint(0,len(ascii_art)) - 1\n"
        '    print("    Get to work!")\n'
        "    print(ascii_art[i])\n"
    )
    check_stmts(code, False)


def test_async_func(check_stmts):
    check_stmts("async def f():\n  pass\n")


def test_async_decorator(check_stmts):
    check_stmts("@g\nasync def f():\n  pass", False)


def test_async_await(check_stmts):
    check_stmts("async def f():\n    await fut\n", False)


# test invalid expressions


def test_syntax_error_del_literal(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("del 7")


def test_syntax_error_del_constant(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("del True")


def test_syntax_error_del_emptytuple(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("del ()")


def test_syntax_error_del_call(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("del foo()")


def test_syntax_error_del_lambda(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string('del lambda x: "yay"')


def test_syntax_error_del_ifexp(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("del x if y else z")


@pytest.mark.parametrize(
    "exp",
    [
        "[i for i in foo]",
        "{i for i in foo}",
        "(i for i in foo)",
        "{k:v for k,v in d.items()}",
    ],
)
def test_syntax_error_del_comps(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"del {exp}")


@pytest.mark.parametrize("exp", ["x + y", "x and y", "-x"])
def test_syntax_error_del_ops(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"del {exp}")


@pytest.mark.parametrize("exp", ["x > y", "x > y == z"])
def test_syntax_error_del_cmp(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"del {exp}")


def test_syntax_error_lonely_del(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("del")


def test_syntax_error_assign_literal(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("7 = x")


def test_syntax_error_assign_constant(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("True = 8")


def test_syntax_error_assign_emptytuple(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("() = x")


def test_syntax_error_assign_call(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("foo() = x")


def test_syntax_error_assign_lambda(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string('lambda x: "yay" = y')


def test_syntax_error_assign_ifexp(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("x if y else z = 8")


@pytest.mark.parametrize(
    "exp",
    [
        "[i for i in foo]",
        "{i for i in foo}",
        "(i for i in foo)",
        "{k:v for k,v in d.items()}",
    ],
)
def test_syntax_error_assign_comps(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"{exp} = z")


@pytest.mark.parametrize("exp", ["x + y", "x and y", "-x"])
def test_syntax_error_assign_ops(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"{exp} = z")


@pytest.mark.parametrize("exp", ["x > y", "x > y == z"])
def test_syntax_error_assign_cmp(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"{exp} = a")


def test_syntax_error_augassign_literal(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("7 += x")


def test_syntax_error_augassign_constant(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("True += 8")


def test_syntax_error_augassign_emptytuple(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("() += x")


def test_syntax_error_augassign_call(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("foo() += x")


def test_syntax_error_augassign_lambda(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string('lambda x: "yay" += y')


def test_syntax_error_augassign_ifexp(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("x if y else z += 8")


@pytest.mark.parametrize(
    "exp",
    [
        "[i for i in foo]",
        "{i for i in foo}",
        "(i for i in foo)",
        "{k:v for k,v in d.items()}",
    ],
)
def test_syntax_error_augassign_comps(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"{exp} += z")


@pytest.mark.parametrize("exp", ["x + y", "x and y", "-x"])
def test_syntax_error_augassign_ops(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"{exp} += z")


@pytest.mark.parametrize("exp", ["x > y", "x > y +=+= z"])
def test_syntax_error_augassign_cmp(parser, exp):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"{exp} += a")


def test_syntax_error_bar_kwonlyargs(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("def spam(*):\n   pass\n", mode="exec")


def test_syntax_error_nondefault_follows_default(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("def spam(x=1, y):\n   pass\n", mode="exec")


def test_syntax_error_lambda_nondefault_follows_default(parser):
    with pytest.raises(SyntaxError):
        parser.parse_string("lambda x=1, y: x", mode="exec")


@pytest.mark.parametrize(
    "first_prefix, second_prefix", itertools.permutations(["", "p", "b"], 2)
)
def test_syntax_error_literal_concat_different(first_prefix, second_prefix, parser):
    with pytest.raises(SyntaxError):
        parser.parse_string(f"{first_prefix}'hello' {second_prefix}'world'")


# match statement
# (tests asserting that pure python match statements produce
# the same ast with the xonsh parser as they do with the python parser)


def test_match_and_case_are_not_keywords(check_stmts):
    check_stmts(
        """
match = 1
case = 2
def match():
    pass
class case():
    pass
"""
    )
