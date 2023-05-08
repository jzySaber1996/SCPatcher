import pyparser

def test_expr_generate(self):
    """test the round trip of expressions to AST back to python source"""
    x = 1
    y = 2

    class F(object):
        def bar(self, a, b):
            return a + b

    def lala(arg):
        return "blah" + arg

    local_dict = dict(x=x, y=y, foo=F(), lala=lala)

    code = "str((x+7*y) / foo.bar(5,6)) + lala('ho')"
    astnode = pyparser.parse(code)
    newcode = pyparser.ExpressionGenerator(astnode).value()
    eq_(eval(code, local_dict), eval(newcode, local_dict))

    a = ["one", "two", "three"]
    hoho = {'somevalue': "asdf"}
    g = [1, 2, 3, 4, 5]
    local_dict = dict(a=a, hoho=hoho, g=g)
    code = "a[2] + hoho['somevalue'] + " \
           "repr(g[3:5]) + repr(g[3:]) + repr(g[:5])"
    astnode = pyparser.parse(code)
    newcode = pyparser.ExpressionGenerator(astnode).value()
    eq_(eval(code, local_dict), eval(newcode, local_dict))

    local_dict = {'f': lambda: 9, 'x': 7}
    code = "x+f()"
    astnode = pyparser.parse(code)
    newcode = pyparser.ExpressionGenerator(astnode).value()
    eq_(eval(code, local_dict), eval(newcode, local_dict))

    for code in ["repr({'x':7,'y':18})",
                 "repr([])",
                 "repr({})",
                 "repr([{3:[]}])",
                 "repr({'x':37*2 + len([6,7,8])})",
                 "repr([1, 2, {}, {'x':'7'}])",
                 "repr({'x':-1})", "repr(((1,2,3), (4,5,6)))",
                 "repr(1 and 2 and 3 and 4)",
                 "repr(True and False or 55)",
                 "repr(lambda x, y: x+y)",
                 "repr(1 & 2 | 3)",
                 "repr(3//5)",
                 "repr(3^5)",
                 "repr([q.endswith('e') for q in "
                 "['one', 'two', 'three']])",
                 "repr([x for x in (5,6,7) if x == 6])",
                 "repr(not False)"]:
        local_dict = {}
        astnode = pyparser.parse(code)
        newcode = pyparser.ExpressionGenerator(astnode).value()
        eq_(eval(code, local_dict),
            eval(newcode, local_dict)
            )