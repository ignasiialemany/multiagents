#!/usr/bin/env python3
"""
Code Documentation Generator
Parses Python source files using AST and generates Markdown documentation.

Usage: python docgen.py input.py -o output.md
"""

import ast
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


# =============================================================================
# EXTRACTOR MODULE - Extracts information from AST nodes
# =============================================================================

class DocEntity:
    """Standardized interface for documentation entities."""
    
    def __init__(self, name: str, docstring: str = "", entity_type: str = ""):
        self.name = name
        self.docstring = docstring
        self.entity_type = entity_type
        self.children: List['DocEntity'] = []
    
    def add_child(self, entity: 'DocEntity'):
        self.children.append(entity)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "docstring": self.docstring,
            "entity_type": self.entity_type,
            "children": [c.to_dict() for c in self.children]
        }


class Parameter:
    """Represents a function/class parameter."""
    
    def __init__(self, name: str, annotation: Optional[str] = None, 
                 default: Optional[str] = None, kind: str = "positional"):
        self.name = name
        self.annotation = annotation
        self.default = default
        self.kind = kind
    
    def to_markdown(self) -> str:
        parts = [self.name]
        if self.annotation:
            parts.append(f": {self.annotation}")
        if self.default:
            parts.append(f" = {self.default}")
        return "".join(parts)


class FunctionExtractor:
    """Extracts documentation from function definitions."""
    
    @staticmethod
    def get_annotation_string(annotation: ast.AST) -> Optional[str]:
        """Convert annotation AST to string."""
        if annotation is None:
            return None
        
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return f"{FunctionExtractor.get_annotation_string(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            base = FunctionExtractor.get_annotation_string(annotation.value)
            subs = FunctionExtractor.get_annotation_string(annotation.slice)
            return f"{base}[{subs}]"
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Tuple):
            elts = [FunctionExtractor.get_annotation_string(e) for e in annotation.elts]
            return ", ".join(elts)
        elif isinstance(annotation, ast.BinOp):
            left = FunctionExtractor.get_annotation_string(annotation.left)
            right = FunctionExtractor.get_annotation_string(annotation.right)
            op = annotation.op.__class__.__name__
            return f"{left} {op} {right}"
        
        return None
    
    @staticmethod
    def get_default_string(default: ast.AST) -> Optional[str]:
        """Convert default value AST to string."""
        if default is None:
            return None
        
        if isinstance(default, ast.Constant):
            if isinstance(default.value, str):
                return f'"{default.value}"'
            return str(default.value)
        elif isinstance(default, ast.Name):
            return default.id
        elif isinstance(default, ast.Tuple):
            elts = [FunctionExtractor.get_default_string(e) for e in default.elts]
            return f"({', '.join(elts)})"
        elif isinstance(default, ast.List):
            elts = [FunctionExtractor.get_default_string(e) for e in default.elts]
            return f"[{', '.join(elts)}]"
        elif isinstance(default, ast.Dict):
            if default.keys:
                items = []
                for k, v in zip(default.keys, default.values):
                    ks = FunctionExtractor.get_default_string(k)
                    vs = FunctionExtractor.get_default_string(v)
                    items.append(f"{ks}: {vs}")
                return f"{{{', '.join(items)}}}"
            return "{}"
        elif isinstance(default, ast.Attribute):
            return f"{FunctionExtractor.get_default_string(default.value)}.{default.attr}"
        elif isinstance(default, ast.BinOp):
            left = FunctionExtractor.get_default_string(default.left)
            right = FunctionExtractor.get_default_string(default.right)
            return f"{left} {default.op.__class__.__name__} {right}"
        
        return str(ast.unparse(default)) if hasattr(ast, 'unparse') else None
    
    @staticmethod
    def extract(node: ast.FunctionDef, parent_name: str = "") -> DocEntity:
        """Extract function documentation from AST node."""
        name = node.name
        
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Create entity
        entity = DocEntity(name, docstring, "function")
        
        # Extract parameters
        args = node.args
        defaults_offset = len(args.args) - len(args.defaults)
        
        for i, arg in enumerate(args.args):
            annotation = None
            if arg.annotation:
                annotation = FunctionExtractor.get_annotation_string(arg.annotation)
            
            default = None
            default_idx = i - defaults_offset
            if default_idx >= 0 and default_idx < len(args.defaults):
                default = FunctionExtractor.get_default_string(args.defaults[default_idx])
            
            # Determine parameter kind
            kind = "positional-or-keyword"
            if arg.arg in ('self', 'cls'):
                kind = "self/cls"
            elif i >= len(args.args) - len(args.kwonlyargs):
                kind = "keyword-only"
            
            param = Parameter(arg.arg, annotation, default, kind)
            entity.add_child(param)
        
        # Handle *args and **kwargs
        if args.vararg:
            annotation = None
            if args.vararg.annotation:
                annotation = FunctionExtractor.get_annotation_string(args.vararg.annotation)
            param = Parameter(f"*{args.vararg.arg}", annotation, None, "varargs")
            entity.add_child(param)
        
        if args.kwarg:
            annotation = None
            if args.kwarg.annotation:
                annotation = FunctionExtractor.get_annotation_string(args.kwarg.annotation)
            param = Parameter(f"**{args.kwarg.arg}", annotation, None, "kwargs")
            entity.add_child(param)
        
        # Extract inner functions/classes
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                child = FunctionExtractor.extract(item, name)
                entity.add_child(child)
            elif isinstance(item, ast.ClassDef):
                child = ClassExtractor.extract(item)
                entity.add_child(child)
        
        return entity


class ClassExtractor:
    """Extracts documentation from class definitions."""
    
    @staticmethod
    def extract(node: ast.ClassDef) -> DocEntity:
        """Extract class documentation from AST node."""
        name = node.name
        
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Create entity
        entity = DocEntity(name, docstring, "class")
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method = FunctionExtractor.extract(item, name)
                entity.add_child(method)
        
        return entity


class ModuleExtractor:
    """Extracts documentation from module (file) level."""
    
    @staticmethod
    def extract(tree: ast.AST, filename: str = "") -> DocEntity:
        """Extract module documentation from AST tree."""
        # Get module docstring
        docstring = ast.get_docstring(tree) or ""
        
        # Create entity
        entity = DocEntity(filename or "module", docstring, "module")
        
        # Extract top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                func = FunctionExtractor.extract(node)
                entity.add_child(func)
            elif isinstance(node, ast.ClassDef):
                cls = ClassExtractor.extract(node)
                entity.add_child(cls)
            elif isinstance(node, ast.Assign):
                # Handle module-level variables
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        try:
                            var_value = ast.unparse(node.value) if hasattr(ast, 'unparse') else "..."
                        except:
                            var_value = "..."
                        var_entity = DocEntity(var_name, "", "variable")
                        var_entity.value = var_value
                        entity.add_child(var_entity)
        
        return entity


# =============================================================================
# FORMATTER MODULE - Renders documentation to Markdown
# =============================================================================

class MarkdownFormatter:
    """Renders DocEntity objects to Markdown format."""
    
    @staticmethod
    def format(entity: DocEntity, level: int = 1) -> str:
        """Format a documentation entity as Markdown."""
        lines = []
        
        # Header based on entity type
        if entity.entity_type == "module":
            lines.append(f"# Module: {entity.name}")
        elif entity.entity_type == "class":
            lines.append(f"## Class: `{entity.name}`")
        elif entity.entity_type == "function":
            lines.append(f"### Function: `{entity.name}()`")
        elif entity.entity_type == "variable":
            lines.append(f"### Variable: `{entity.name}`")
        
        # Docstring
        if entity.docstring:
            lines.append("")
            lines.append(entity.docstring)
        
        # Parameters (for functions)
        params = [c for c in entity.children if isinstance(c, Parameter)]
        if params:
            lines.append("")
            lines.append("**Parameters:**")
            lines.append("")
            for param in params:
                param_str = param.to_markdown()
                if param.annotation:
                    lines.append(f"- `{param_str}`")
                else:
                    lines.append(f"- `{param.name}`" + (f" (default: `{param.default}`)" if param.default else ""))
        
        # Children (methods, inner classes, etc.)
        for child in entity.children:
            if not isinstance(child, Parameter):
                child_md = MarkdownFormatter.format(child, level + 1)
                if child_md:
                    lines.append("")
                    lines.append(child_md)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_module(entity: DocEntity) -> str:
        """Format a complete module as Markdown."""
        lines = []
        
        # Title
        if entity.docstring:
            lines.append(f"# {entity.name}")
            lines.append("")
            lines.append(entity.docstring)
        else:
            lines.append(f"# {entity.name}")
        
        lines.append("")
        
        # Separate children by type
        classes = [c for c in entity.children if c.entity_type == "class"]
        functions = [c for c in entity.children if c.entity_type == "function"]
        variables = [c for c in entity.children if c.entity_type == "variable"]
        
        # Classes section
        if classes:
            lines.append("## Classes")
            lines.append("")
            for cls in classes:
                cls_md = MarkdownFormatter.format(cls)
                lines.append(cls_md)
                lines.append("")
        
        # Functions section
        if functions:
            lines.append("## Functions")
            lines.append("")
            for func in functions:
                func_md = MarkdownFormatter.format(func)
                lines.append(func_md)
                lines.append("")
        
        # Variables section
        if variables:
            lines.append("## Module Variables")
            lines.append("")
            for var in variables:
                lines.append(f"### `{var.name}`")
                if hasattr(var, 'value'):
                    lines.append(f"```python\n{var.name} = {var.value}\n```")
                lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown documentation from Python source files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python docgen.py input.py -o docs.md
  python docgen.py input.py --stdout
  python docgen.py input.py -o docs.md --title "My Documentation"
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input Python file to document"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output Markdown file path",
        default=None
    )
    
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output to stdout instead of file"
    )
    
    parser.add_argument(
        "--title",
        help="Custom title for the documentation",
        default=None
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.suffix == ".py":
        print(f"Error: Input file must be a Python file (.py)", file=sys.stderr)
        sys.exit(1)
    
    # Parse the Python file
    if args.verbose:
        print(f"Parsing: {input_path}")
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        tree = ast.parse(source_code, filename=str(input_path))
    except SyntaxError as e:
        print(f"Error: Failed to parse '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract documentation
    if args.verbose:
        print("Extracting documentation...")
    
    module_name = input_path.stem
    module_entity = ModuleExtractor.extract(tree, module_name)
    
    # Generate Markdown
    if args.verbose:
        print("Generating Markdown...")
    
    markdown = MarkdownFormatter.format_module(module_entity)
    
    # Add custom title if provided
    if args.title:
        markdown = f"# {args.title}\n\n" + markdown
    
    # Output
    if args.stdout or args.output is None:
        print(markdown)
    else:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        if args.verbose:
            print(f"Documentation written to: {output_path}")
    
    print("Documentation generated successfully!")


# =============================================================================
# TEST / SAMPLE FUNCTIONALITY
# =============================================================================

def run_demo():
    """Run a demonstration of the documentation generator."""
    # Sample Python code for demonstration
    sample_code = '''"""Sample module for demonstration.

This module contains various Python constructs to demonstrate
the documentation generator's capabilities.
"""

import typing
from typing import List, Dict, Optional


class DataProcessor:
    """A class for processing data.
    
    This class demonstrates how class documentation is extracted.
    It can process various types of data and return results.
    """
    
    def __init__(self, name: str, max_size: int = 100):
        """Initialize the DataProcessor.
        
        Args:
            name: The name of the processor.
            max_size: Maximum size of data to process.
        """
        self.name = name
        self.max_size = max_size
        self.processed_count = 0
    
    def process(self, data: List[int], multiplier: float = 1.0) -> List[float]:
        """Process a list of data points.
        
        Args:
            data: List of integers to process.
            multiplier: Multiplication factor.
            
        Returns:
            List of processed floats.
        """
        result = [x * multiplier for x in data[:self.max_size]]
        self.processed_count += len(result)
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics.
        
        Returns:
            Dictionary with statistics.
        """
        return {
            "name": self.name,
            "processed": self.processed_count,
            "max_size": self.max_size
        }


def calculate_sum(numbers: List[float], *, precision: int = 2) -> float:
    """Calculate the sum of numbers with precision.
    
    Args:
        numbers: List of numbers to sum.
        precision: Decimal precision (keyword-only).
        
    Returns:
        Sum as a float.
    """
    return round(sum(numbers), precision)


def greet(name: str = "World", greeting: str = "Hello") -> str:
    """Generate a greeting message.
    
    Args:
        name: Name to greet.
        greeting: Greeting word.
        
    Returns:
        Formatted greeting string.
    """
    return f"{greeting}, {name}!"


# Module variable example
MAX_BUFFER_SIZE = 1024
'''
    
    # Parse and generate docs
    tree = ast.parse(sample_code)
    module_entity = ModuleExtractor.extract(tree, "sample_module")
    markdown = MarkdownFormatter.format_module(module_entity)
    
    print("=" * 60)
    print("DOCUMENTATION GENERATOR DEMO")
    print("=" * 60)
    print("\nGENERATED DOCUMENTATION:")
    print("-" * 60)
    print(markdown)
    print("-" * 60)


if __name__ == "__main__":
    # Check if we should run demo or main
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--demo"):
        run_demo()
    else:
        main()