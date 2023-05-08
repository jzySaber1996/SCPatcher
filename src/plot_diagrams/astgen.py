import ast

code = ast.parse("print('Hello world!')")
print(code)
exec(compile(code, filename="", mode="exec"))