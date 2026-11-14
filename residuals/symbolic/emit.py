"""Code emitters for generated residuals.

wrenfold derives and simplifies the expressions; these generators only decide how
the result is spelled. Every target gets flat row-major pointer arguments, so the
generated code needs no header beyond the target's own math functions:

- C++ (CPU backend): `template <typename Scalar>` functions over raw pointers.
- Metal Shading Language: the same body over `thread` pointers, with `metal::`
  math functions. Instantiated with `float` inside kernels.

Optional outputs are pointers that may be null, so a caller that only needs the
residual skips the Jacobian work.
"""

from wrenfold import ast, code_generation as cg, type_info


def _flat_size(arg) -> int:
    t = arg.type
    if isinstance(t, type_info.MatrixType):
        return t.rows * t.cols
    return 1


def _flat_index(arg, row: int, col: int) -> int:
    return row * arg.type.cols + col


class _PointerCppBase(cg.CppGenerator):
    """Shared formatting for C-like targets that take flat arrays."""

    pointer_prefix = ""  # address-space qualifier, e.g. "thread " in Metal
    std_namespace = "std"
    inline_keyword = "inline"

    def _param(self, arg) -> str:
        if arg.is_input:
            if isinstance(arg.type, type_info.MatrixType):
                return f"{self.pointer_prefix}const Scalar* {arg.name}"
            return f"const Scalar {arg.name}"
        return f"{self.pointer_prefix}Scalar* {arg.name}"

    def format_function_signature(self, sig: ast.FunctionSignature) -> str:
        params = ", ".join(self._param(a) for a in sig.arguments)
        return f"template <typename Scalar>\n{self.inline_keyword} void {sig.name}({params})"

    def format_function_definition(self, element: ast.FunctionDefinition) -> str:
        body = self._format_statements(element.body, indent=1)
        return f"{self.format(element.signature)}\n{{\n{body}\n}}\n"

    def _format_statements(self, statements, indent: int) -> str:
        pad = "  " * indent
        lines = []
        for s in statements:
            text = self.format(s)
            lines.extend(pad + line if line else line for line in text.split("\n"))
        return "\n".join(lines)

    def format_get_argument(self, element: ast.GetArgument) -> str:
        return element.argument.name

    def format_get_matrix_element(self, element: ast.GetMatrixElement) -> str:
        arg = element.arg.argument
        return f"{arg.name}[{_flat_index(arg, element.row, element.col)}]"

    def format_assign_output_matrix(self, element: ast.AssignOutputMatrix) -> str:
        arg = element.arg
        values = element.value.args
        assert len(values) == _flat_size(arg)
        return "\n".join(f"{arg.name}[{i}] = {self.format(v)};" for i, v in enumerate(values))

    def format_assign_output_scalar(self, element: ast.AssignOutputScalar) -> str:
        return f"*{element.arg.name} = {self.format(element.value)};"

    def format_optional_output_branch(self, element: ast.OptionalOutputBranch) -> str:
        body = self._format_statements(element.statements, indent=1)
        return f"if ({element.argument.name} != nullptr) {{\n{body}\n}}"

    def format_call_std_function(self, element: ast.CallStdFunction) -> str:
        text = super().format_call_std_function(element)
        if self.std_namespace != "std":
            text = text.replace("std::", f"{self.std_namespace}::")
        return text


class CpuGenerator(_PointerCppBase):
    pass


class MetalGenerator(_PointerCppBase):
    pointer_prefix = "thread "
    std_namespace = "metal"


def generate(func, generator, name=None) -> str:
    return cg.generate_function(func, generator=generator, name=name)
