"""Tokenization help for Xonsh programs.

tokenize(readline) is a generator that breaks a stream of bytes into
Python tokens.  It decodes the bytes according to PEP-0263 for
determining source file encoding.

It accepts a readline-like method which is called repeatedly to get the
next line of input (or b"" for EOF).  It generates 5-tuples with these
members:

    the token type (see token.py)
    the token (a string)
    the starting (row, column) indices of the token (a 2-tuple of ints)
    the ending (row, column) indices of the token (a 2-tuple of ints)
    the original line (string)

Original file credits:
   __author__ = 'Ka-Ping Yee <ping@lfw.org>'
   __credits__ = ('GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, '
                  'Skip Montanaro, Raymond Hettinger, Trent Nelson, '
                  'Michael Foord')
"""

import builtins
import codecs
import collections
import functools
import io
import itertools
import re

from .tokens import *

cookie_re = re.compile(r"^[ \t\f]*#.*coding[:=][ \t]*([-\w.]+)", re.ASCII)
blank_re = re.compile(rb"^[ \t\f]*(?:[#\r\n]|$)", re.ASCII)

__all__ = [
    "tokenize",
    "detect_encoding",
    "untokenize",
    "TokenInfo",
    "TokenError",
]

AUGASSIGN_OPS = r"[+\-*/%&@|^=<>:]=?"


additional_parenlevs = frozenset({"@(", "!(", "![", "$(", "$[", "${", "@$("})


class TokenInfo(collections.namedtuple("TokenInfo", "type string start end line")):
    def __repr__(self):
        annotated_type = "%d (%s)" % (self.type, tok_name[self.type])
        return (
            "TokenInfo(type={}, string={!r}, start={!r}, end={!r}, line={!r})".format(
                *self._replace(type=annotated_type)
            )
        )

    @property
    def exact_type(self):
        if self.type == OP and self.string in EXACT_TOKEN_TYPES:
            return EXACT_TOKEN_TYPES[self.string]
        else:
            return self.type

    @property
    def name(self):
        return tok_name[self.exact_type]


def group(*choices):
    return "(" + "|".join(choices) + ")"


def tokany(*choices):
    return group(*choices) + "*"


def maybe(*choices):
    return group(*choices) + "?"


# Note: we use unicode matching for names ("\w") but ascii matching for
# number literals.
Whitespace = r"[ \f\t]*"
Comment = r"#[^\r\n]*"
Ignore = Whitespace + tokany(r"\\\r?\n" + Whitespace) + maybe(Comment)
Name = r"\$?\w+"

Hexnumber = r"0[xX](?:_?[0-9a-fA-F])+"
Binnumber = r"0[bB](?:_?[01])+"
Octnumber = r"0[oO](?:_?[0-7])+"
Decnumber = r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)"
Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
Exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
Pointfloat = group(
    r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*"
) + maybe(Exponent)
Expfloat = r"[0-9](?:_?[0-9])*" + Exponent
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group(r"[0-9](?:_?[0-9])*[jJ]", Floatnumber + r"[jJ]")
Number = group(Imagnumber, Floatnumber, Intnumber)


# Return the empty string, plus all of the valid string prefixes.
def _all_string_prefixes():
    # The valid string prefixes. Only contain the lower case versions,
    #  and don't contain any permutations (include 'fr', but not
    #  'rf'). The various permutations will be generated.
    _valid_string_prefixes = ["b", "r", "u", "f", "br", "fr", "p", "pf", "pr"]
    # if we add binary f-strings, add: ['fb', 'fbr']
    result = {""}
    for prefix in _valid_string_prefixes:
        for t in itertools.permutations(prefix):
            # create a list with upper and lower versions of each
            #  character
            for u in itertools.product(*[(c, c.upper()) for c in t]):
                result.add("".join(u))
    return result


@functools.lru_cache
def _compile(expr):
    return re.compile(expr, re.UNICODE)


# Note that since _all_string_prefixes includes the empty string,
#  StringPrefix can be the empty string (making it optional).
StringPrefix = group(*_all_string_prefixes())

# Tail end of ' string.
Single = r"[^'\\]*(?:\\.[^'\\]*)*'"
# Tail end of " string.
Double = r'[^"\\]*(?:\\.[^"\\]*)*"'
# Tail end of ''' string.
Single3 = r"[^'\\]*(?:(?:\\.|'(?!''))[^'\\]*)*'''"
# Tail end of """ string.
Double3 = r'[^"\\]*(?:(?:\\.|"(?!""))[^"\\]*)*"""'
Triple = group(StringPrefix + "'''", StringPrefix + '"""')
# Single-line ' or " string.
String = group(
    StringPrefix + r"'[^\n'\\]*(?:\\.[^\n'\\]*)*'",
    StringPrefix + r'"[^\n"\\]*(?:\\.[^\n"\\]*)*"',
)

# Xonsh-specific Syntax
SearchPath = r"((?:[rgpf]+|@\w*)?)`([^\n`\\]*(?:\\.[^\n`\\]*)*)`"

# Because of leftmost-then-longest match semantics, be sure to put the
# longest operators first (e.g., if = came before ==, == would get
# recognized as two instances of =).
_redir_names = ("out", "all", "err", "e", "2", "a", "&", "1", "o")
_redir_map = (
    # stderr to stdout
    "err>out",
    "err>&1",
    "2>out",
    "err>o",
    "err>1",
    "e>out",
    "e>&1",
    "2>&1",
    "e>o",
    "2>o",
    "e>1",
    "2>1",
    # stdout to stderr
    "out>err",
    "out>&2",
    "1>err",
    "out>e",
    "out>2",
    "o>err",
    "o>&2",
    "1>&2",
    "o>e",
    "1>e",
    "o>2",
    "1>2",
)
IORedirect = group(group(*_redir_map), f"{group(*_redir_names)}>>?")

_redir_check_0 = set(_redir_map)
_redir_check_1 = {f"{i}>" for i in _redir_names}.union(_redir_check_0)
_redir_check_2 = {f"{i}>>" for i in _redir_names}.union(_redir_check_1)
_redir_check = frozenset(_redir_check_2)

Operator = group(
    r"\*\*=?",
    r">>=?",
    r"<<=?",
    r"!=",
    r"//=?",
    r"->",
    r"@\$\(?",
    r"\|\|",
    "&&",
    r"@\(",
    r"!\(",
    r"!\[",
    r"\$\(",
    r"\$\[",
    r"\${",
    r"\?\?",
    r"\?",
    AUGASSIGN_OPS,
    r"~",
)

Bracket = "[][(){}]"
Special = group(r"\r?\n", r"\.\.\.", r"[:;.,@]")
Funny = group(Operator, Bracket, Special)

PlainToken = group(IORedirect, Number, Funny, String, Name, SearchPath)

# First (or only) line of ' or " string.
ContStr = group(
    StringPrefix + r"'[^\n'\\]*(?:\\.[^\n'\\]*)*" + group("'", r"\\\r?\n"),
    StringPrefix + r'"[^\n"\\]*(?:\\.[^\n"\\]*)*' + group('"', r"\\\r?\n"),
)
PseudoExtras = group(r"\\\r?\n|\Z", Comment, Triple, SearchPath)
PseudoToken = Whitespace + group(PseudoExtras, IORedirect, Number, Funny, ContStr, Name)

# For a given string prefix plus quotes, endpats maps it to a regex
#  to match the remainder of that string. _prefix can be empty, for
#  a normal single or triple quoted string (with no prefix).
endpats = {}
for _prefix in _all_string_prefixes():
    endpats[_prefix + "'"] = Single
    endpats[_prefix + '"'] = Double
    endpats[_prefix + "'''"] = Single3
    endpats[_prefix + '"""'] = Double3
del _prefix

# A set of all of the single and triple quoted string prefixes,
#  including the opening quotes.
single_quoted = set()
triple_quoted = set()
for t in _all_string_prefixes():
    for u in (t + '"', t + "'"):
        single_quoted.add(u)
    for u in (t + '"""', t + "'''"):
        triple_quoted.add(u)
del t, u

tabsize = 8


class TokenError(Exception):
    pass


class StopTokenizing(Exception):
    pass


class Untokenizer:
    def __init__(self):
        self.tokens = []
        self.prev_row = 1
        self.prev_col = 0
        self.encoding = None

    def add_whitespace(self, start):
        row, col = start
        if row < self.prev_row or row == self.prev_row and col < self.prev_col:
            raise ValueError(
                "start ({},{}) precedes previous end ({},{})".format(
                    row, col, self.prev_row, self.prev_col
                )
            )
        row_offset = row - self.prev_row
        if row_offset:
            self.tokens.append("\\\n" * row_offset)
            self.prev_col = 0
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(" " * col_offset)

    def untokenize(self, iterable):
        it = iter(iterable)
        indents = []
        startline = False
        for t in it:
            if len(t) == 2:
                self.compat(t, it)
                break
            tok_type, token, start, end, line = t
            if tok_type == ENCODING:
                self.encoding = token
                continue
            if tok_type == ENDMARKER:
                break
            if tok_type == INDENT:
                indents.append(token)
                continue
            elif tok_type == DEDENT:
                indents.pop()
                self.prev_row, self.prev_col = end
                continue
            elif tok_type in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                indent = indents[-1]
                if start[1] >= len(indent):
                    self.tokens.append(indent)
                    self.prev_col = len(indent)
                startline = False
            self.add_whitespace(start)
            self.tokens.append(token)
            self.prev_row, self.prev_col = end
            if tok_type in (NEWLINE, NL):
                self.prev_row += 1
                self.prev_col = 0
        return "".join(self.tokens)

    def compat(self, token, iterable):
        indents = []
        toks_append = self.tokens.append
        startline = token[0] in (NEWLINE, NL)
        prevstring = False

        for tok in itertools.chain([token], iterable):
            toknum, tokval = tok[:2]
            if toknum == ENCODING:
                self.encoding = tokval
                continue

            if toknum in (NAME, NUMBER):
                tokval += " "

            # Insert a space between two consecutive strings
            if toknum == STRING:
                if prevstring:
                    tokval = " " + tokval
                prevstring = True
            else:
                prevstring = False

            if toknum == INDENT:
                indents.append(tokval)
                continue
            elif toknum == DEDENT:
                indents.pop()
                continue
            elif toknum in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                toks_append(indents[-1])
                startline = False
            toks_append(tokval)


def untokenize(iterable):
    """Transform tokens back into Python source code.
    It returns a bytes object, encoded using the ENCODING
    token, which is the first token sequence output by tokenize.

    Each element returned by the iterable must be a token sequence
    with at least two elements, a token number and token value.  If
    only two tokens are passed, the resulting output is poor.

    Round-trip invariant for full input:
        Untokenized source will match input source exactly

    Round-trip invariant for limited input:
        # Output bytes will tokenize back to the input
        t1 = [tok[:2] for tok in tokenize(f.readline)]
        newcode = untokenize(t1)
        readline = BytesIO(newcode).readline
        t2 = [tok[:2] for tok in tokenize(readline)]
        assert t1 == t2
    """
    ut = Untokenizer()
    out = ut.untokenize(iterable)
    if ut.encoding is not None:
        out = out.encode(ut.encoding)
    return out


def _get_normal_name(orig_enc):
    """Imitates get_normal_name in tokenizer.c."""
    # Only care about the first 12 characters.
    enc = orig_enc[:12].lower().replace("_", "-")
    if enc == "utf-8" or enc.startswith("utf-8-"):
        return "utf-8"
    if enc in ("latin-1", "iso-8859-1", "iso-latin-1") or enc.startswith(
        ("latin-1-", "iso-8859-1-", "iso-latin-1-")
    ):
        return "iso-8859-1"
    return orig_enc


def detect_encoding(readline):
    """
    The detect_encoding() function is used to detect the encoding that should
    be used to decode a Python source file.  It requires one argument, readline,
    in the same way as the tokenize() generator.

    It will call readline a maximum of twice, and return the encoding used
    (as a string) and a list of any lines (left as bytes) it has read in.

    It detects the encoding from the presence of a utf-8 bom or an encoding
    cookie as specified in pep-0263.  If both a bom and a cookie are present,
    but disagree, a SyntaxError will be raised.  If the encoding cookie is an
    invalid charset, raise a SyntaxError.  Note that if a utf-8 bom is found,
    'utf-8-sig' is returned.

    If no encoding is specified, then the default of 'utf-8' will be returned.
    """
    try:
        filename = readline.__self__.name
    except AttributeError:
        filename = None
    bom_found = False
    encoding = None
    default = "utf-8"

    def read_or_stop():
        try:
            return readline()
        except StopIteration:
            return b""

    def find_cookie(line):
        try:
            # Decode as UTF-8. Either the line is an encoding declaration,
            # in which case it should be pure ASCII, or it must be UTF-8
            # per default encoding.
            line_string = line.decode("utf-8")
        except UnicodeDecodeError:
            msg = "invalid or missing encoding declaration"
            if filename is not None:
                msg = f"{msg} for {filename!r}"
            raise SyntaxError(msg)

        match = cookie_re.match(line_string)
        if not match:
            return None
        encoding = _get_normal_name(match.group(1))
        try:
            codecs.lookup(encoding)
        except LookupError:
            # This behaviour mimics the Python interpreter
            if filename is None:
                msg = "unknown encoding: " + encoding
            else:
                msg = f"unknown encoding for {filename!r}: {encoding}"
            raise SyntaxError(msg)

        if bom_found:
            if encoding != "utf-8":
                # This behaviour mimics the Python interpreter
                if filename is None:
                    msg = "encoding problem: utf-8"
                else:
                    msg = f"encoding problem for {filename!r}: utf-8"
                raise SyntaxError(msg)
            encoding += "-sig"
        return encoding

    first = read_or_stop()
    if first.startswith(codecs.BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = "utf-8-sig"
    if not first:
        return default, []

    encoding = find_cookie(first)
    if encoding:
        return encoding, [first]
    if not blank_re.match(first):
        return default, [first]

    second = read_or_stop()
    if not second:
        return default, [first]

    encoding = find_cookie(second)
    if encoding:
        return encoding, [first, second]

    return default, [first, second]


def tokopen(filename):
    """Open a file in read only mode using the encoding detected by
    detect_encoding().
    """
    buffer = builtins.open(filename, "rb")
    try:
        encoding, lines = detect_encoding(buffer.readline)
        buffer.seek(0)
        text = io.TextIOWrapper(buffer, encoding, line_buffering=True)
        text.mode = "r"
        return text
    except Exception:
        buffer.close()
        raise


def tokenize(readline, tolerant=False):
    """
    The tokenize() generator requires one argument, readline, which
    must be a callable object which provides the same interface as the
    readline() method of built-in file objects.  Each call to the function
    should return one line of input as bytes.  Alternately, readline
    can be a callable function terminating with StopIteration:
        readline = open(myfile, 'rb').__next__  # Example of alternate readline

    The generator produces 5-tuples with these members: the token type; the
    token string; a 2-tuple (srow, scol) of ints specifying the row and
    column where the token begins in the source; a 2-tuple (erow, ecol) of
    ints specifying the row and column where the token ends in the source;
    and the line on which the token was found.  The line passed is the
    logical line; continuation lines are included.

    The first token sequence will always be an ENCODING token
    which tells you which encoding was used to decode the bytes stream.

    If ``tolerant`` is True, yield ERRORTOKEN with the erroneous string instead of
    throwing an exception when encountering an error.
    """
    encoding, consumed = detect_encoding(readline)
    rl_gen = iter(readline, b"")
    empty = itertools.repeat(b"")
    return _tokenize(
        itertools.chain(consumed, rl_gen, empty).__next__, encoding, tolerant
    )


def _tokenize(readline, encoding, tolerant=False):
    lnum = parenlev = continued = 0
    numchars = "0123456789"
    contstr, needcont = "", 0
    contline = None
    indents = [0]

    if encoding is not None:
        if encoding == "utf-8-sig":
            # BOM will already have been stripped.
            encoding = "utf-8"
        yield TokenInfo(ENCODING, encoding, (0, 0), (0, 0), "")
    last_line = b""
    line = b""
    while True:  # loop over lines in stream
        try:
            # We capture the value of the line variable here because
            # readline uses the empty string '' to signal end of input,
            # hence `line` itself will always be overwritten at the end
            # of this loop.
            last_line = line
            line = readline()
        except StopIteration:
            line = b""

        if encoding is not None:
            line = line.decode(encoding)
        lnum += 1
        pos, max = 0, len(line)

        if contstr:  # continued string
            if not line:
                if tolerant:
                    # return the partial string
                    yield TokenInfo(
                        ERRORTOKEN, contstr, strstart, (lnum, end), contline + line
                    )
                    break
                else:
                    raise TokenError("EOF in multi-line string", strstart)
            endmatch = endprog.match(line)
            if endmatch:
                pos = end = endmatch.end(0)
                yield TokenInfo(
                    STRING, contstr + line[:end], strstart, (lnum, end), contline + line
                )
                contstr, needcont = "", 0
                contline = None
            elif needcont and line[-2:] != "\\\n" and line[-3:] != "\\\r\n":
                yield TokenInfo(
                    ERRORTOKEN, contstr + line, strstart, (lnum, len(line)), contline
                )
                contstr = ""
                contline = None
                continue
            else:
                contstr = contstr + line
                contline = contline + line
                continue

        elif parenlev == 0 and not continued:  # new statement
            if not line:
                break
            column = 0
            while pos < max:  # measure leading whitespace
                if line[pos] == " ":
                    column += 1
                elif line[pos] == "\t":
                    column = (column // tabsize + 1) * tabsize
                elif line[pos] == "\f":
                    column = 0
                else:
                    break
                pos += 1
            if pos == max:
                break

            if line[pos] in "#\r\n":  # skip comments or blank lines
                if line[pos] == "#":
                    comment_token = line[pos:].rstrip("\r\n")
                    yield TokenInfo(
                        COMMENT,
                        comment_token,
                        (lnum, pos),
                        (lnum, pos + len(comment_token)),
                        line,
                    )
                    pos += len(comment_token)

                yield TokenInfo(NL, line[pos:], (lnum, pos), (lnum, len(line)), line)
                continue

            if column > indents[-1]:  # count indents or dedents
                indents.append(column)
                yield TokenInfo(INDENT, line[:pos], (lnum, 0), (lnum, pos), line)
            while column < indents[-1]:
                if (
                    column not in indents and not tolerant
                ):  # if tolerant, just ignore the error
                    raise IndentationError(
                        "unindent does not match any outer indentation level",
                        ("<tokenize>", lnum, pos, line),
                    )
                indents = indents[:-1]

                yield TokenInfo(DEDENT, "", (lnum, pos), (lnum, pos), line)

        else:  # continued statement
            if not line:
                if tolerant:
                    # no need to raise an error, we're done
                    break
                raise TokenError("EOF in multi-line statement", (lnum, 0))
            continued = 0

        while pos < max:
            pseudomatch = _compile(PseudoToken).match(line, pos)
            if pseudomatch:  # scan for tokens
                start, end = pseudomatch.span(1)
                spos, epos, pos = (lnum, start), (lnum, end), end
                if start == end:
                    continue
                token, initial = line[start:end], line[start]

                if token in _redir_check:
                    yield TokenInfo(IOREDIRECT, token, spos, epos, line)
                elif initial in numchars or (  # ordinary number
                    initial == "." and token != "." and token != "..."
                ):
                    yield TokenInfo(NUMBER, token, spos, epos, line)
                elif initial in "\r\n":
                    if parenlev > 0:
                        yield TokenInfo(NL, token, spos, epos, line)
                    else:
                        yield TokenInfo(NEWLINE, token, spos, epos, line)

                elif initial == "#":
                    assert not token.endswith("\n")
                    yield TokenInfo(COMMENT, token, spos, epos, line)
                # Xonsh-specific Regex Globbing
                elif re.match(SearchPath, token):
                    yield TokenInfo(SEARCHPATH, token, spos, epos, line)
                elif token in triple_quoted:
                    endprog = _compile(endpats[token])
                    endmatch = endprog.match(line, pos)
                    if endmatch:  # all on one line
                        pos = endmatch.end(0)
                        token = line[start:pos]
                        yield TokenInfo(STRING, token, spos, (lnum, pos), line)
                    else:
                        strstart = (lnum, start)  # multiple lines
                        contstr = line[start:]
                        contline = line
                        break

                # Check up to the first 3 chars of the token to see if
                #  they're in the single_quoted set. If so, they start
                #  a string.
                # We're using the first 3, because we're looking for
                #  "rb'" (for example) at the start of the token. If
                #  we switch to longer prefixes, this needs to be
                #  adjusted.
                # Note that initial == token[:1].
                # Also note that single quote checking must come after
                #  triple quote checking (above).
                elif (
                    initial in single_quoted
                    or token[:2] in single_quoted
                    or token[:3] in single_quoted
                ):
                    if token[-1] == "\n":  # continued string
                        strstart = (lnum, start)
                        # Again, using the first 3 chars of the
                        #  token. This is looking for the matching end
                        #  regex for the correct type of quote
                        #  character. So it's really looking for
                        #  endpats["'"] or endpats['"'], by trying to
                        #  skip string prefix characters, if any.
                        endprog = _compile(
                            endpats.get(initial)
                            or endpats.get(token[1])
                            or endpats.get(token[2])
                        )
                        contstr, needcont = line[start:], 1
                        contline = line
                        break
                    else:  # ordinary string
                        yield TokenInfo(STRING, token, spos, epos, line)
                elif token.startswith("$") and token[1:].isidentifier():
                    yield TokenInfo(DOLLARNAME, token, spos, epos, line)
                elif initial.isidentifier():  # ordinary name
                    yield TokenInfo(NAME, token, spos, epos, line)
                # elif token == "\\\n" or token == "\\\r\n":  # continued stmt
                #     continued = 1
                #     yield TokenInfo(ERRORTOKEN, token, spos, epos, line)
                elif initial == "\\":  # continued stmt
                    # for cases like C:\\path\\to\\file
                    continued = 1
                else:
                    if initial in "([{":
                        parenlev += 1
                    elif initial in ")]}":
                        parenlev -= 1
                    elif token in additional_parenlevs:
                        parenlev += 1
                    yield TokenInfo(OP, token, spos, epos, line)
            else:
                yield TokenInfo(
                    ERRORTOKEN, line[pos], (lnum, pos), (lnum, pos + 1), line
                )
                pos += 1

    # Add an implicit NEWLINE if the input doesn't end in one
    if (
        last_line
        and last_line[-1] not in "\r\n"
        and not last_line.strip().startswith("#")
    ):
        yield TokenInfo(
            NEWLINE, "", (lnum - 1, len(last_line)), (lnum - 1, len(last_line) + 1), ""
        )
    for indent in indents[1:]:  # pop remaining indent levels
        yield TokenInfo(DEDENT, "", (lnum, 0), (lnum, 0), "")
    yield TokenInfo(ENDMARKER, "", (lnum, 0), (lnum, 0), "")


# An undocumented, backwards compatible, API for all the places in the standard
# library that expect to be able to use tokenize with strings
def generate_tokens(readline, *, tolerant=False):
    return _tokenize(readline, encoding=None, tolerant=tolerant)
