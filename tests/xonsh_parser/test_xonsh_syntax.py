import itertools
import textwrap
from ast import AST, Call, Pass, Str, With, JoinedStr, Expression

import pytest


def test_f_env_var(check_xonsh_ast):
    check_xonsh_ast({}, 'f"{$HOME}"', run=False)
    check_xonsh_ast({}, "f'{$XONSH_DEBUG}'", run=False)
    check_xonsh_ast({}, 'F"{$PATH} and {$XONSH_DEBUG}"', run=False)


fstring_adaptor_parameters = [
    ('f"$HOME"', "$HOME"),
    ('f"{0} - {1}"', "0 - 1"),
    ('f"{$HOME}"', "/foo/bar"),
    ('f"{ $HOME }"', "/foo/bar"),
    ("f\"{'$HOME'}\"", "$HOME"),
    ('f"$HOME  = {$HOME}"', "$HOME  = /foo/bar"),
    ("f\"{${'HOME'}}\"", "/foo/bar"),
    ("f'{${$FOO+$BAR}}'", "/foo/bar"),
    ("f\"${$FOO}{$BAR}={f'{$HOME}'}\"", "$HOME=/foo/bar"),
    (
        '''f"""foo
{f"_{$HOME}_"}
bar"""''',
        "foo\n_/foo/bar_\nbar",
    ),
    (
        '''f"""foo
{f"_{${'HOME'}}_"}
bar"""''',
        "foo\n_/foo/bar_\nbar",
    ),
    (
        '''f"""foo
{f"_{${ $FOO + $BAR }}_"}
bar"""''',
        "foo\n_/foo/bar_\nbar",
    ),
    ("f'{$HOME=}'", "$HOME='/foo/bar'"),
]


@pytest.mark.parametrize("inp, exp", fstring_adaptor_parameters)
@pytest.mark.xfail
def test_fstring_adaptor(inp, xsh, exp, monkeypatch):
    joined_str_node = FStringAdaptor(inp, "f").run()
    assert isinstance(joined_str_node, JoinedStr)
    node = Expression(body=joined_str_node)
    code = compile(node, "<test_fstring_adaptor>", mode="eval")
    xenv = {"HOME": "/foo/bar", "FOO": "HO", "BAR": "ME"}
    for key, val in xenv.items():
        monkeypatch.setitem(xsh.env, key, val)
    obs = eval(code)
    assert exp == obs


def test_path_literal(check_xonsh_ast):
    check_xonsh_ast({}, 'p"/foo"', False)
    check_xonsh_ast({}, 'pr"/foo"', False)
    check_xonsh_ast({}, 'rp"/foo"', False)
    check_xonsh_ast({}, 'pR"/foo"', False)
    check_xonsh_ast({}, 'Rp"/foo"', False)


def test_path_fstring_literal(check_xonsh_ast):
    check_xonsh_ast({}, 'pf"/foo"', False)
    check_xonsh_ast({}, 'fp"/foo"', False)
    check_xonsh_ast({}, 'pF"/foo"', False)
    check_xonsh_ast({}, 'Fp"/foo"', False)
    check_xonsh_ast({}, 'pf"/foo{1+1}"', False)
    check_xonsh_ast({}, 'fp"/foo{1+1}"', False)
    check_xonsh_ast({}, 'pF"/foo{1+1}"', False)
    check_xonsh_ast({}, 'Fp"/foo{1+1}"', False)


@pytest.mark.parametrize(
    "first_prefix, second_prefix",
    itertools.product(["p", "pf", "pr"], repeat=2),
)
def test_path_literal_concat(first_prefix, second_prefix, check_xonsh_ast):
    check_xonsh_ast(
        {}, first_prefix + r"'11{a}22\n'" + " " + second_prefix + r"'33{b}44\n'", False
    )


def test_dollar_name(unparse):
    assert unparse("$WAKKA") == "__xonsh__.env['WAKKA']"


@pytest.mark.xfail
def test_dollar_py(unparse):
    assert unparse('x = "WAKKA"; y = ${x}') == ""


@pytest.mark.xfail
def test_dollar_py_test(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": 42}, '${None or "WAKKA"}')


@pytest.mark.xfail
def test_dollar_py_recursive_name(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": 42, "JAWAKA": "WAKKA"}, "${$JAWAKA}")


@pytest.mark.xfail
def test_dollar_py_test_recursive_name(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": 42, "JAWAKA": "WAKKA"}, "${None or $JAWAKA}")


@pytest.mark.xfail
def test_dollar_py_test_recursive_test(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": 42, "JAWAKA": "WAKKA"}, '${${"JAWA" + $JAWAKA[-2:]}}')


@pytest.mark.xfail
def test_dollar_name_set(check_xonsh):
    check_xonsh({"WAKKA": 42}, "$WAKKA = 42")


@pytest.mark.xfail
def test_dollar_py_set(check_xonsh):
    check_xonsh({"WAKKA": 42}, 'x = "WAKKA"; ${x} = 65')


def test_dollar_sub(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls)", False)


@pytest.mark.parametrize(
    "expr",
    [
        "$(ls )",
        "$( ls)",
        "$( ls )",
    ],
)
def test_dollar_sub_space(expr, check_xonsh_ast):
    check_xonsh_ast({}, expr, False)


def test_ls_dot(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls .)", False)


def test_lambda_in_atparens(check_xonsh_ast):
    check_xonsh_ast(
        {}, '$(echo hello | @(lambda a, s=None: "hey!") foo bar baz)', False
    )


def test_generator_in_atparens(check_xonsh_ast):
    check_xonsh_ast({}, "$(echo @(i**2 for i in range(20)))", False)


def test_bare_tuple_in_atparens(check_xonsh_ast):
    check_xonsh_ast({}, '$(echo @("a", 7))', False)


def test_nested_madness(check_xonsh_ast):
    check_xonsh_ast(
        {},
        "$(@$(which echo) ls "
        "| @(lambda a, s=None: $(@(s.strip()) @(a[1]))) foo -la baz)",
        False,
    )


def test_atparens_intoken(check_xonsh_ast):
    check_xonsh_ast({}, "![echo /x/@(y)/z]", False)


def test_ls_dot_nesting(check_xonsh_ast):
    check_xonsh_ast({}, '$(ls @(None or "."))', False)


def test_ls_dot_nesting_var(check_xonsh):
    check_xonsh({}, 'x = "."; $(ls @(None or x))', False)


def test_ls_dot_str(check_xonsh_ast):
    check_xonsh_ast({}, '$(ls ".")', False)


def test_ls_nest_ls(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls $(ls))", False)


def test_ls_nest_ls_dashl(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls $(ls) -l)", False)


def test_ls_envvar_strval(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": "."}, "$(ls $WAKKA)", False)


def test_ls_envvar_listval(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": [".", "."]}, "$(ls $WAKKA)", False)


def test_bang_sub(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls)", False)


@pytest.mark.parametrize(
    "expr",
    [
        "!(ls )",
        "!( ls)",
        "!( ls )",
    ],
)
def test_bang_sub_space(expr, check_xonsh_ast):
    check_xonsh_ast({}, expr, False)


def test_bang_ls_dot(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls .)", False)


def test_bang_ls_dot_nesting(check_xonsh_ast):
    check_xonsh_ast({}, '!(ls @(None or "."))', False)


def test_bang_ls_dot_nesting_var(check_xonsh):
    check_xonsh({}, 'x = "."; !(ls @(None or x))', False)


def test_bang_ls_dot_str(check_xonsh_ast):
    check_xonsh_ast({}, '!(ls ".")', False)


def test_bang_ls_nest_ls(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls $(ls))", False)


def test_bang_ls_nest_ls_dashl(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls $(ls) -l)", False)


def test_bang_ls_envvar_strval(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": "."}, "!(ls $WAKKA)", False)


def test_bang_ls_envvar_listval(check_xonsh_ast):
    check_xonsh_ast({"WAKKA": [".", "."]}, "!(ls $WAKKA)", False)


def test_bang_envvar_args(check_xonsh_ast):
    check_xonsh_ast({"LS": "ls"}, "!($LS .)", False)


@pytest.mark.xfail
def test_question(check_xonsh_ast):
    check_xonsh_ast({}, "range?")


@pytest.mark.xfail
def test_dobquestion(check_xonsh_ast):
    check_xonsh_ast({}, "range??")


@pytest.mark.xfail
def test_question_chain(check_xonsh_ast):
    check_xonsh_ast({}, "range?.index?")


def test_ls_regex(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls `[Ff]+i*LE` -l)", False)


@pytest.mark.parametrize("p", ["", "p"])
@pytest.mark.parametrize("f", ["", "f"])
@pytest.mark.parametrize("glob_type", ["", "r", "g"])
def test_backtick(p, f, glob_type, check_xonsh_ast):
    check_xonsh_ast({}, f"print({p}{f}{glob_type}`.*`)", False)


def test_ls_regex_octothorpe(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls `#[Ff]+i*LE` -l)", False)


def test_ls_explicitregex(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls r`[Ff]+i*LE` -l)", False)


def test_ls_explicitregex_octothorpe(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls r`#[Ff]+i*LE` -l)", False)


def test_ls_glob(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls g`[Ff]+i*LE` -l)", False)


def test_ls_glob_octothorpe(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls g`#[Ff]+i*LE` -l)", False)


def test_ls_customsearch(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls @foo`[Ff]+i*LE` -l)", False)


def test_custombacktick(check_xonsh_ast):
    check_xonsh_ast({}, "print(@foo`.*`)", False)


def test_ls_customsearch_octothorpe(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls @foo`#[Ff]+i*LE` -l)", False)


def test_injection(check_xonsh_ast):
    check_xonsh_ast({}, "$[@$(which python)]", False)


def test_rhs_nested_injection(check_xonsh_ast):
    check_xonsh_ast({}, "$[ls @$(dirname @$(which python))]", False)


def test_merged_injection(check_xonsh_ast):
    tree = check_xonsh_ast({}, "![a@$(echo 1 2)b]", False, return_obs=True)
    assert isinstance(tree, AST)
    func = tree.body.args[0].right.func
    assert func.attr == "list_of_list_of_strs_outer_product"


def test_backtick_octothorpe(check_xonsh_ast):
    check_xonsh_ast({}, "print(`#.*`)", False)


def test_uncaptured_sub(check_xonsh_ast):
    check_xonsh_ast({}, "$[ls]", False)


def test_hiddenobj_sub(check_xonsh_ast):
    check_xonsh_ast({}, "![ls]", False)


def test_slash_envarv_echo(check_xonsh_ast):
    check_xonsh_ast({}, "![echo $HOME/place]", False)


def test_echo_double_eq(check_xonsh_ast):
    check_xonsh_ast({}, "![echo yo==yo]", False)


def test_bang_two_cmds_one_pipe(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls | grep wakka)", False)


def test_bang_three_cmds_two_pipes(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls | grep wakka | grep jawaka)", False)


def test_bang_one_cmd_write(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls > x.py)", False)


def test_bang_one_cmd_append(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls >> x.py)", False)


def test_bang_two_cmds_write(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls | grep wakka > x.py)", False)


def test_bang_two_cmds_append(check_xonsh_ast):
    check_xonsh_ast({}, "!(ls | grep wakka >> x.py)", False)


def test_bang_cmd_background(check_xonsh_ast):
    check_xonsh_ast({}, "!(emacs ugggh &)", False)


def test_bang_cmd_background_nospace(check_xonsh_ast):
    check_xonsh_ast({}, "!(emacs ugggh&)", False)


def test_bang_git_quotes_no_space(check_xonsh_ast):
    check_xonsh_ast({}, '![git commit -am "wakka"]', False)


def test_bang_git_quotes_space(check_xonsh_ast):
    check_xonsh_ast({}, '![git commit -am "wakka jawaka"]', False)


def test_bang_git_two_quotes_space(check_xonsh):
    check_xonsh(
        {},
        '![git commit -am "wakka jawaka"]\n' '![git commit -am "flock jawaka"]\n',
        False,
    )


def test_bang_git_two_quotes_space_space(check_xonsh):
    check_xonsh(
        {},
        '![git commit -am "wakka jawaka" ]\n'
        '![git commit -am "flock jawaka milwaka" ]\n',
        False,
    )


def test_bang_ls_quotes_3_space(check_xonsh_ast):
    check_xonsh_ast({}, '![ls "wakka jawaka baraka"]', False)


def test_two_cmds_one_pipe(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls | grep wakka)", False)


def test_three_cmds_two_pipes(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls | grep wakka | grep jawaka)", False)


def test_two_cmds_one_and_brackets(check_xonsh_ast):
    check_xonsh_ast({}, "![ls me] and ![grep wakka]", False)


def test_three_cmds_two_ands(check_xonsh_ast):
    check_xonsh_ast({}, "![ls] and ![grep wakka] and ![grep jawaka]", False)


def test_two_cmds_one_doubleamps(check_xonsh_ast):
    check_xonsh_ast({}, "![ls] && ![grep wakka]", False)


def test_three_cmds_two_doubleamps(check_xonsh_ast):
    check_xonsh_ast({}, "![ls] && ![grep wakka] && ![grep jawaka]", False)


def test_two_cmds_one_or(check_xonsh_ast):
    check_xonsh_ast({}, "![ls] or ![grep wakka]", False)


def test_three_cmds_two_ors(check_xonsh_ast):
    check_xonsh_ast({}, "![ls] or ![grep wakka] or ![grep jawaka]", False)


def test_two_cmds_one_doublepipe(check_xonsh_ast):
    check_xonsh_ast({}, "![ls] || ![grep wakka]", False)


def test_three_cmds_two_doublepipe(check_xonsh_ast):
    check_xonsh_ast({}, "![ls] || ![grep wakka] || ![grep jawaka]", False)


def test_one_cmd_write(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls > x.py)", False)


def test_one_cmd_append(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls >> x.py)", False)


def test_two_cmds_write(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls | grep wakka > x.py)", False)


def test_two_cmds_append(check_xonsh_ast):
    check_xonsh_ast({}, "$(ls | grep wakka >> x.py)", False)


def test_cmd_background(check_xonsh_ast):
    check_xonsh_ast({}, "$(emacs ugggh &)", False)


def test_cmd_background_nospace(check_xonsh_ast):
    check_xonsh_ast({}, "$(emacs ugggh&)", False)


def test_git_quotes_no_space(check_xonsh_ast):
    check_xonsh_ast({}, '$[git commit -am "wakka"]', False)


def test_git_quotes_space(check_xonsh_ast):
    check_xonsh_ast({}, '$[git commit -am "wakka jawaka"]', False)


def test_git_two_quotes_space(check_xonsh):
    check_xonsh(
        {},
        '$[git commit -am "wakka jawaka"]\n' '$[git commit -am "flock jawaka"]\n',
        False,
    )


def test_git_two_quotes_space_space(check_xonsh):
    check_xonsh(
        {},
        '$[git commit -am "wakka jawaka" ]\n'
        '$[git commit -am "flock jawaka milwaka" ]\n',
        False,
    )


def test_ls_quotes_3_space(check_xonsh_ast):
    check_xonsh_ast({}, '$[ls "wakka jawaka baraka"]', False)


def test_leading_envvar_assignment(check_xonsh_ast):
    check_xonsh_ast({}, "![$FOO='foo' $BAR=2 echo r'$BAR']", False)


def test_echo_comma(check_xonsh_ast):
    check_xonsh_ast({}, "![echo ,]", False)


def test_echo_internal_comma(check_xonsh_ast):
    check_xonsh_ast({}, "![echo 1,2]", False)


def test_comment_only(check_xonsh_ast):
    check_xonsh_ast({}, "# hello")


def test_echo_slash_question(check_xonsh_ast):
    check_xonsh_ast({}, "![echo /?]", False)


def test_bad_quotes(check_xonsh_ast):
    with pytest.raises(SyntaxError):
        check_xonsh_ast({}, '![echo """hello]', False)


def test_redirect(check_xonsh_ast):
    assert check_xonsh_ast({}, "$[cat < input.txt]", False)
    assert check_xonsh_ast({}, "$[< input.txt cat]", False)


@pytest.mark.parametrize(
    "case",
    [
        "![(cat)]",
        "![(cat;)]",
        "![(cd path; ls; cd)]",
        '![(echo "abc"; sleep 1; echo "def")]',
        '![(echo "abc"; sleep 1; echo "def") | grep abc]',
        "![(if True:\n   ls\nelse:\n   echo not true)]",
    ],
)
def test_use_subshell(case, check_xonsh_ast):
    check_xonsh_ast({}, case, False, debug_level=0)


@pytest.mark.parametrize(
    "case",
    [
        "$[cat < /path/to/input.txt]",
        "$[(cat) < /path/to/input.txt]",
        "$[< /path/to/input.txt cat]",
        "![< /path/to/input.txt]",
        "![< /path/to/input.txt > /path/to/output.txt]",
    ],
)
def test_redirect_abspath(case, check_xonsh_ast):
    assert check_xonsh_ast({}, case, False)


@pytest.mark.parametrize("case", ["", "o", "out", "1"])
def test_redirect_output(case, check_xonsh_ast):
    assert check_xonsh_ast({}, f'$[echo "test" {case}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[< input.txt echo "test" {case}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[echo "test" {case}> test.txt < input.txt]', False)


@pytest.mark.parametrize("case", ["e", "err", "2"])
def test_redirect_error(case, check_xonsh_ast):
    assert check_xonsh_ast({}, f'$[echo "test" {case}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[< input.txt echo "test" {case}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[echo "test" {case}> test.txt < input.txt]', False)


@pytest.mark.parametrize("case", ["a", "all", "&"])
def test_redirect_all(case, check_xonsh_ast):
    assert check_xonsh_ast({}, f'$[echo "test" {case}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[< input.txt echo "test" {case}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[echo "test" {case}> test.txt < input.txt]', False)


@pytest.mark.parametrize(
    "r",
    [
        "e>o",
        "e>out",
        "err>o",
        "2>1",
        "e>1",
        "err>1",
        "2>out",
        "2>o",
        "err>&1",
        "e>&1",
        "2>&1",
    ],
)
@pytest.mark.parametrize("o", ["", "o", "out", "1"])
def test_redirect_error_to_output(r, o, check_xonsh_ast):
    assert check_xonsh_ast({}, f'$[echo "test" {r} {o}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[< input.txt echo "test" {r} {o}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[echo "test" {r} {o}> test.txt < input.txt]', False)


@pytest.mark.parametrize(
    "r",
    [
        "o>e",
        "o>err",
        "out>e",
        "1>2",
        "o>2",
        "out>2",
        "1>err",
        "1>e",
        "out>&2",
        "o>&2",
        "1>&2",
    ],
)
@pytest.mark.parametrize("e", ["e", "err", "2"])
def test_redirect_output_to_error(r, e, check_xonsh_ast):
    assert check_xonsh_ast({}, f'$[echo "test" {r} {e}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[< input.txt echo "test" {r} {e}> test.txt]', False)
    assert check_xonsh_ast({}, f'$[echo "test" {r} {e}> test.txt < input.txt]', False)


def test_macro_call_empty(check_xonsh_ast):
    assert check_xonsh_ast({}, "f!()", False)


MACRO_ARGS = [
    "x",
    "True",
    "None",
    "import os",
    "x=10",
    '"oh no, mom"',
    "...",
    " ... ",
    "if True:\n  pass",
    "{x: y}",
    "{x: y, 42: 5}",
    "{1, 2, 3,}",
    "(x,y)",
    "(x, y)",
    "((x, y), z)",
    "g()",
    "range(10)",
    "range(1, 10, 2)",
    "()",
    "{}",
    "[]",
    "[1, 2]",
    "@(x)",
    "!(ls -l)",
    "![ls -l]",
    "$(ls -l)",
    "${x + y}",
    "$[ls -l]",
    "@$(which xonsh)",
]


@pytest.mark.parametrize("s", MACRO_ARGS)
def test_macro_call_one_arg(check_xonsh_ast, s):
    f = f"f!({s})"
    tree = check_xonsh_ast({}, f, False, return_obs=True)
    assert isinstance(tree, AST)
    args = tree.body.args[1].elts
    assert len(args) == 1
    assert args[0].s == s.strip()


@pytest.mark.parametrize("s,t", itertools.product(MACRO_ARGS[::2], MACRO_ARGS[1::2]))
def test_macro_call_two_args(check_xonsh_ast, s, t):
    f = f"f!({s}, {t})"
    tree = check_xonsh_ast({}, f, False, return_obs=True)
    assert isinstance(tree, AST)
    args = tree.body.args[1].elts
    assert len(args) == 2
    assert args[0].s == s.strip()
    assert args[1].s == t.strip()


@pytest.mark.parametrize(
    "s,t,u", itertools.product(MACRO_ARGS[::3], MACRO_ARGS[1::3], MACRO_ARGS[2::3])
)
def test_macro_call_three_args(check_xonsh_ast, s, t, u):
    f = f"f!({s}, {t}, {u})"
    tree = check_xonsh_ast({}, f, False, return_obs=True)
    assert isinstance(tree, AST)
    args = tree.body.args[1].elts
    assert len(args) == 3
    assert args[0].s == s.strip()
    assert args[1].s == t.strip()
    assert args[2].s == u.strip()


@pytest.mark.parametrize("s", MACRO_ARGS)
def test_macro_call_one_trailing(check_xonsh_ast, s):
    f = f"f!({s},)"
    tree = check_xonsh_ast({}, f, False, return_obs=True)
    assert isinstance(tree, AST)
    args = tree.body.args[1].elts
    assert len(args) == 1
    assert args[0].s == s.strip()


@pytest.mark.parametrize("s", MACRO_ARGS)
def test_macro_call_one_trailing_space(check_xonsh_ast, s):
    f = f"f!( {s}, )"
    tree = check_xonsh_ast({}, f, False, return_obs=True)
    assert isinstance(tree, AST)
    args = tree.body.args[1].elts
    assert len(args) == 1
    assert args[0].s == s.strip()


SUBPROC_MACRO_OC = [("!(", ")"), ("$(", ")"), ("![", "]"), ("$[", "]")]


@pytest.mark.parametrize("opener, closer", SUBPROC_MACRO_OC)
@pytest.mark.parametrize("body", ["echo!", "echo !", "echo ! "])
def test_empty_subprocbang(opener, closer, body, check_xonsh_ast):
    tree = check_xonsh_ast({}, opener + body + closer, False, return_obs=True)
    assert isinstance(tree, AST)
    cmd = tree.body.args[0].elts
    assert len(cmd) == 2
    assert cmd[1].s == ""


@pytest.mark.parametrize("opener, closer", SUBPROC_MACRO_OC)
@pytest.mark.parametrize("body", ["echo!x", "echo !x", "echo !x", "echo ! x"])
def test_single_subprocbang(opener, closer, body, check_xonsh_ast):
    tree = check_xonsh_ast({}, opener + body + closer, False, return_obs=True)
    assert isinstance(tree, AST)
    cmd = tree.body.args[0].elts
    assert len(cmd) == 2
    assert cmd[1].s == "x"


@pytest.mark.parametrize("opener, closer", SUBPROC_MACRO_OC)
@pytest.mark.parametrize(
    "body", ["echo -n!x", "echo -n!x", "echo -n !x", "echo -n ! x"]
)
def test_arg_single_subprocbang(opener, closer, body, check_xonsh_ast):
    tree = check_xonsh_ast({}, opener + body + closer, False, return_obs=True)
    assert isinstance(tree, AST)
    cmd = tree.body.args[0].elts
    assert len(cmd) == 3
    assert cmd[2].s == "x"


@pytest.mark.parametrize("opener, closer", SUBPROC_MACRO_OC)
@pytest.mark.parametrize("ipener, iloser", [("$(", ")"), ("@$(", ")"), ("$[", "]")])
@pytest.mark.parametrize(
    "body", ["echo -n!x", "echo -n!x", "echo -n !x", "echo -n ! x"]
)
def test_arg_single_subprocbang_nested(
    opener, closer, ipener, iloser, body, check_xonsh_ast
):
    tree = check_xonsh_ast({}, opener + body + closer, False, return_obs=True)
    assert isinstance(tree, AST)
    cmd = tree.body.args[0].elts
    assert len(cmd) == 3
    assert cmd[2].s == "x"


@pytest.mark.parametrize("opener, closer", SUBPROC_MACRO_OC)
@pytest.mark.parametrize(
    "body",
    [
        "echo!x + y",
        "echo !x + y",
        "echo !x + y",
        "echo ! x + y",
        "timeit! bang! and more",
        "timeit! recurse() and more",
        "timeit! recurse[] and more",
        "timeit! recurse!() and more",
        "timeit! recurse![] and more",
        "timeit! recurse$() and more",
        "timeit! recurse$[] and more",
        "timeit! recurse!() and more",
        "timeit!!!!",
        "timeit! (!)",
        "timeit! [!]",
        "timeit!!(ls)",
        'timeit!"!)"',
    ],
)
def test_many_subprocbang(opener, closer, body, check_xonsh_ast):
    tree = check_xonsh_ast({}, opener + body + closer, False, return_obs=True)
    assert isinstance(tree, AST)
    cmd = tree.body.args[0].elts
    assert len(cmd) == 2
    assert cmd[1].s == body.partition("!")[-1].strip()


WITH_BANG_RAWSUITES = [
    "pass\n",
    "x = 42\ny = 12\n",
    'export PATH="yo:momma"\necho $PATH\n',
    ("with q as t:\n" "    v = 10\n" "\n"),
    (
        "with q as t:\n"
        "    v = 10\n"
        "\n"
        "for x in range(6):\n"
        "    if True:\n"
        "        pass\n"
        "    else:\n"
        "        ls -l\n"
        "\n"
        "a = 42\n"
    ),
]


@pytest.mark.parametrize("body", WITH_BANG_RAWSUITES)
def test_withbang_single_suite(body, check_xonsh_ast):
    code = "with! x:\n{}".format(textwrap.indent(body, "    "))
    tree = check_xonsh_ast({}, code, False, return_obs=True, mode="exec")
    assert isinstance(tree, AST)
    wither = tree.body[0]
    assert isinstance(wither, With)
    assert len(wither.body) == 1
    assert isinstance(wither.body[0], Pass)
    assert len(wither.items) == 1
    item = wither.items[0]
    s = item.context_expr.args[1].s
    assert s == body


@pytest.mark.parametrize("body", WITH_BANG_RAWSUITES)
def test_withbang_as_single_suite(body, check_xonsh_ast):
    code = "with! x as y:\n{}".format(textwrap.indent(body, "    "))
    tree = check_xonsh_ast({}, code, False, return_obs=True, mode="exec")
    assert isinstance(tree, AST)
    wither = tree.body[0]
    assert isinstance(wither, With)
    assert len(wither.body) == 1
    assert isinstance(wither.body[0], Pass)
    assert len(wither.items) == 1
    item = wither.items[0]
    assert item.optional_vars.id == "y"
    s = item.context_expr.args[1].s
    assert s == body


@pytest.mark.parametrize("body", WITH_BANG_RAWSUITES)
def test_withbang_single_suite_trailing(body, check_xonsh_ast):
    code = "with! x:\n{}\nprint(x)\n".format(textwrap.indent(body, "    "))
    tree = check_xonsh_ast(
        {},
        code,
        False,
        return_obs=True,
        mode="exec",
        # debug_level=100
    )
    assert isinstance(tree, AST)
    wither = tree.body[0]
    assert isinstance(wither, With)
    assert len(wither.body) == 1
    assert isinstance(wither.body[0], Pass)
    assert len(wither.items) == 1
    item = wither.items[0]
    s = item.context_expr.args[1].s
    assert s == body + "\n"


WITH_BANG_RAWSIMPLE = [
    "pass",
    "x = 42; y = 12",
    'export PATH="yo:momma"; echo $PATH',
    "[1,\n    2,\n    3]",
]


@pytest.mark.parametrize("body", WITH_BANG_RAWSIMPLE)
def test_withbang_single_simple(body, check_xonsh_ast):
    code = f"with! x: {body}\n"
    tree = check_xonsh_ast({}, code, False, return_obs=True, mode="exec")
    assert isinstance(tree, AST)
    wither = tree.body[0]
    assert isinstance(wither, With)
    assert len(wither.body) == 1
    assert isinstance(wither.body[0], Pass)
    assert len(wither.items) == 1
    item = wither.items[0]
    s = item.context_expr.args[1].s
    assert s == body


@pytest.mark.parametrize("body", WITH_BANG_RAWSIMPLE)
def test_withbang_single_simple_opt(body, check_xonsh_ast):
    code = f"with! x as y: {body}\n"
    tree = check_xonsh_ast({}, code, False, return_obs=True, mode="exec")
    assert isinstance(tree, AST)
    wither = tree.body[0]
    assert isinstance(wither, With)
    assert len(wither.body) == 1
    assert isinstance(wither.body[0], Pass)
    assert len(wither.items) == 1
    item = wither.items[0]
    assert item.optional_vars.id == "y"
    s = item.context_expr.args[1].s
    assert s == body


@pytest.mark.parametrize("body", WITH_BANG_RAWSUITES)
def test_withbang_as_many_suite(body, check_xonsh_ast):
    code = "with! x as a, y as b, z as c:\n{}"
    code = code.format(textwrap.indent(body, "    "))
    tree = check_xonsh_ast({}, code, False, return_obs=True, mode="exec")
    assert isinstance(tree, AST)
    wither = tree.body[0]
    assert isinstance(wither, With)
    assert len(wither.body) == 1
    assert isinstance(wither.body[0], Pass)
    assert len(wither.items) == 3
    for i, targ in enumerate("abc"):
        item = wither.items[i]
        assert item.optional_vars.id == targ
        s = item.context_expr.args[1].s
        assert s == body


def test_subproc_raw_str_literal(check_xonsh_ast):
    tree = check_xonsh_ast({}, "!(echo '$foo')", run=False, return_obs=True)
    assert isinstance(tree, AST)
    subproc = tree.body
    assert isinstance(subproc.args[0].elts[1], Call)
    assert subproc.args[0].elts[1].func.attr == "expand_path"

    tree = check_xonsh_ast({}, "!(echo r'$foo')", run=False, return_obs=True)
    assert isinstance(tree, AST)
    subproc = tree.body
    assert isinstance(subproc.args[0].elts[1], Str)
    assert subproc.args[0].elts[1].s == "$foo"


def test_get_repo_url(parser):
    parser.parse(
        "def get_repo_url():\n"
        "    raw = $(git remote get-url --push origin).rstrip()\n"
        "    return raw.replace('https://github.com/', '')\n"
    )
