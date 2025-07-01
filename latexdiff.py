#!/usr/bin/env python3
"""
Python implementation of latexdiff using mdiff for better move detection.

latexdiff - differences two latex files on the word level
            and produces a latex file with the differences marked up.

This is a Python port of the original Perl latexdiff script, enhanced with
better move detection using the mdiff package's diff_lines_with_similarities function.

Copyright (C) 2025 - Python port
Original Copyright (C) 2004-22  F J Tilmann (tilmann@gfz-potsdam.de)
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from mdiff import diff_lines_with_similarities

__version__ = "2.0.0"  # Python port version
__original_version__ = "1.3.5a"  # Based on Perl version


@dataclass
class Config:
    """Configuration class to hold all latexdiff options"""

    type: str = "UNDERLINE"
    subtype: str = "SAFE"
    floattype: str = "FLOATSAFE"
    math_markup: str = "FINE"
    graphics_markup: str = "NEWONLY"
    encoding: str = "utf8"
    verbose: bool = False
    debug: bool = False
    no_del: bool = False
    allow_spaces: bool = False
    flatten: bool = False
    preamble_file: Optional[str] = None
    packages: Optional[List[str]] = None
    labels: Optional[List[str]] = None

    def __post_init__(self):
        if self.packages is None:
            self.packages = []
        if self.labels is None:
            self.labels = []


class LaTeXEnvironmentParser:
    """Parser that identifies and preserves LaTeX environments and commands"""

    def __init__(self):
        # Math environments that should never be split
        self.math_environments = {
            "equation",
            "equation*",
            "align",
            "align*",
            "alignat",
            "alignat*",
            "gather",
            "gather*",
            "multline",
            "multline*",
            "flalign",
            "flalign*",
            "eqnarray",
            "eqnarray*",
            "displaymath",
            "math",
            "array",
            "matrix",
            "pmatrix",
            "bmatrix",
            "vmatrix",
            "Vmatrix",
            "smallmatrix",
            "cases",
            "split",
            "subequations",
        }

        # Block environments that should be treated as units
        self.block_environments = {
            "figure",
            "figure*",
            "table",
            "table*",
            "tabular",
            "tabularx",
            "longtable",
            "theorem",
            "proof",
            "definition",
            "lemma",
            "corollary",
            "example",
            "remark",
            "note",
            "verbatim",
            "lstlisting",
            "minted",
            "algorithmic",
            "algorithm",
            "center",
            "flushleft",
            "flushright",
            "quote",
            "quotation",
            "itemize",
            "enumerate",
            "description",
        }

        # Commands that should not be split
        self.protected_commands = {
            "ref",
            "label",
            "cite",
            "citep",
            "citet",
            "bibliography",
            "bibliographystyle",
            "include",
            "input",
            "usepackage",
            "documentclass",
            "newcommand",
            "renewcommand",
            "def",
            "gdef",
            "edef",
            "xdef",
            "section",
            "subsection",
            "subsubsection",
            "chapter",
            "part",
            "paragraph",
            "subparagraph",
            "caption",
            "title",
            "author",
        }

    def parse_into_blocks(self, text: str) -> List[Dict]:
        """Parse text into logical blocks that preserve LaTeX structure"""
        blocks = []
        i = 0
        current_block = ""

        while i < len(text):
            # Check for environment start
            env_match = re.match(r"\\begin\{([^}]+)\}", text[i:])
            if env_match:
                # Flush current block if not empty
                if current_block.strip():
                    blocks.append(
                        {"type": "text", "content": current_block, "protected": False}
                    )
                    current_block = ""

                # Find the complete environment
                env_name = env_match.group(1)
                env_start = i
                env_content, env_end = self._extract_environment(text, i, env_name)

                # Determine if this environment should be protected
                is_protected = (
                    env_name in self.math_environments
                    or env_name in self.block_environments
                )

                blocks.append(
                    {
                        "type": "environment",
                        "content": env_content,
                        "env_name": env_name,
                        "protected": is_protected,
                    }
                )

                i = env_end
                continue

            # Check for math delimiters
            if text[i : i + 2] == "$$":
                # Flush current block
                if current_block.strip():
                    blocks.append(
                        {"type": "text", "content": current_block, "protected": False}
                    )
                    current_block = ""

                # Find end of display math
                math_start = i
                i += 2
                while i < len(text) - 1:
                    if text[i : i + 2] == "$$":
                        i += 2
                        break
                    i += 1

                blocks.append(
                    {"type": "math", "content": text[math_start:i], "protected": True}
                )
                continue

            elif text[i] == "$":
                # Flush current block
                if current_block.strip():
                    blocks.append(
                        {"type": "text", "content": current_block, "protected": False}
                    )
                    current_block = ""

                # Find end of inline math
                math_start = i
                i += 1
                while i < len(text):
                    if text[i] == "$" and (i == 0 or text[i - 1] != "\\"):
                        i += 1
                        break
                    i += 1

                blocks.append(
                    {"type": "math", "content": text[math_start:i], "protected": True}
                )
                continue

            # Check for commands
            cmd_match = re.match(
                r"\\([a-zA-Z*]+)(?:\[[^\]]*\])?(?:\{[^}]*\})*", text[i:]
            )
            if cmd_match:
                cmd_name = cmd_match.group(1)
                if cmd_name in self.protected_commands:
                    # Flush current block
                    if current_block.strip():
                        blocks.append(
                            {
                                "type": "text",
                                "content": current_block,
                                "protected": False,
                            }
                        )
                        current_block = ""

                    # Add protected command
                    cmd_content = cmd_match.group(0)
                    blocks.append(
                        {"type": "command", "content": cmd_content, "protected": True}
                    )

                    i += len(cmd_content)
                    continue

            # Regular character - add to current block
            current_block += text[i]
            i += 1

        # Add final block
        if current_block.strip():
            blocks.append(
                {"type": "text", "content": current_block, "protected": False}
            )

        return blocks

    def _extract_environment(
        self, text: str, start: int, env_name: str
    ) -> Tuple[str, int]:
        """Extract complete environment from start position"""
        i = start
        depth = 0

        while i < len(text):
            # Look for \begin{env_name}
            begin_match = re.match(rf"\\begin\{{{re.escape(env_name)}\}}", text[i:])
            if begin_match:
                depth += 1
                i += len(begin_match.group(0))
                continue

            # Look for \end{env_name}
            end_match = re.match(rf"\\end\{{{re.escape(env_name)}\}}", text[i:])
            if end_match:
                depth -= 1
                if depth == 0:
                    # Found the matching end
                    i += len(end_match.group(0))
                    return text[start:i], i
                i += len(end_match.group(0))
                continue

            i += 1

        # If we reach here, environment was not properly closed
        return text[start:], len(text)

    def split_text_block(self, text: str) -> List[str]:
        """Split a text block into smaller units while preserving LaTeX structure"""
        # Split by paragraphs (double newlines) and sentence boundaries
        paragraphs = re.split(r"\n\s*\n", text)

        result = []
        for para in paragraphs:
            if not para.strip():
                continue

            # Further split by sentences, but be careful with LaTeX commands
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
            result.extend(sentences)

        return [s for s in result if s.strip()]


class LaTeXDiff:
    """Main LaTeX diff processor with environment awareness"""

    def __init__(self, config: Config):
        self.config = config
        self.parser = LaTeXEnvironmentParser()

        # LaTeX diff markup commands
        self.markup_commands = {
            "add": r"\DIFadd{",
            "del": r"\DIFdel{",
            "move_from": r"\DIFmovefrom{",
            "move_to": r"\DIFmoveto{",
        }

        # Preamble additions for markup
        self.diff_preamble = self._generate_diff_preamble()

    def _generate_diff_preamble(self) -> str:
        """Generate the preamble additions for diff markup"""
        preamble_file = os.path.join(os.path.dirname(__file__), "diff_preamble.tex")
        if os.path.exists(preamble_file):
            with open(preamble_file, "r") as f:
                return f.read()
        else:
            # Fallback preamble
            return """
% DIF PREAMBLE
\\RequirePackage[normalem]{ulem}
\\RequirePackage{color}
\\definecolor{RED}{rgb}{1,0,0}
\\definecolor{BLUE}{rgb}{0,0,1}
\\definecolor{GREEN}{rgb}{0,0.5,0}
\\definecolor{ORANGE}{rgb}{1,0.5,0}
\\providecommand{\\DIFadd}[1]{{\\protect\\color{blue}\\uwave{#1}}}
\\providecommand{\\DIFdel}[1]{{\\protect\\color{red}\\sout{#1}}}
\\providecommand{\\DIFmovefrom}[1]{{\\protect\\color{orange}\\xout{#1}}}
\\providecommand{\\DIFmoveto}[1]{{\\protect\\color{green}\\uwave{#1}}}
% END DIF PREAMBLE
"""

    def process_files(self, old_file: str, new_file: str) -> str:
        """Process two LaTeX files and return the diff markup"""
        if self.config.verbose:
            print(f"Processing {old_file} -> {new_file}", file=sys.stderr)

        # Read files
        old_content = self._read_file(old_file)
        new_content = self._read_file(new_file)

        # Split into preamble and body
        old_preamble, old_body = self._split_document(old_content)
        new_preamble, new_body = self._split_document(new_content)

        # Process preamble differences
        diff_preamble = self._process_preamble_diff(old_preamble, new_preamble)

        # Use environment-aware diff processing
        diff_body = self._generate_environment_aware_diff(old_body, new_body)

        # Combine results
        result = self._combine_diff_document(
            diff_preamble, diff_body, old_file, new_file
        )

        return result

    def _read_file(self, filename: str) -> str:
        """Read file with proper encoding"""
        try:
            with open(filename, "r", encoding=self.config.encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            with open(filename, "r", encoding="latin-1") as f:
                return f.read()

    def _split_document(self, content: str) -> Tuple[str, str]:
        """Split LaTeX document into preamble and body"""
        begin_doc_match = re.search(r"\\begin\{document\}", content)
        if begin_doc_match:
            preamble = content[: begin_doc_match.start()]
            body = content[begin_doc_match.start() :]
            return preamble, body
        else:
            # No \begin{document} found, treat entire content as body
            return "", content

    def _process_preamble_diff(self, old_preamble: str, new_preamble: str) -> str:
        """Process preamble differences"""
        # For now, use new preamble and add diff commands
        result = new_preamble
        if result and not result.endswith("\n"):
            result += "\n"
        result += self.diff_preamble + "\n"
        return result

    def _generate_environment_aware_diff(self, old_body: str, new_body: str) -> str:
        """Generate diff with full LaTeX environment awareness"""

        # Parse into logical blocks
        old_blocks = self.parser.parse_into_blocks(old_body)
        new_blocks = self.parser.parse_into_blocks(new_body)

        if self.config.debug:
            print(f"Old blocks: {len(old_blocks)}", file=sys.stderr)
            print(f"New blocks: {len(new_blocks)}", file=sys.stderr)

        # Convert blocks to strings for comparison
        old_strings = []
        new_strings = []

        for block in old_blocks:
            if block["protected"]:
                # Protected blocks are compared as single units
                old_strings.append(block["content"])
            else:
                # Text blocks can be split further
                sub_blocks = self.parser.split_text_block(block["content"])
                old_strings.extend(sub_blocks)

        for block in new_blocks:
            if block["protected"]:
                # Protected blocks are compared as single units
                new_strings.append(block["content"])
            else:
                # Text blocks can be split further
                sub_blocks = self.parser.split_text_block(block["content"])
                new_strings.extend(sub_blocks)

        # Use mdiff for comparison
        try:
            old_text = "\n".join(old_strings)
            new_text = "\n".join(new_strings)

            a_lines, b_lines, opcodes = diff_lines_with_similarities(
                old_text,
                new_text,
                cutoff=0.6,
                keepends=True,
                case_sensitive=True,
            )

            if self.config.debug:
                print(f"Found {len(opcodes)} opcodes", file=sys.stderr)

            return self._process_opcodes_safely(a_lines, b_lines, opcodes)

        except Exception as e:
            if self.config.debug:
                print(
                    f"mdiff failed: {e}, falling back to simple diff", file=sys.stderr
                )
            return self._simple_diff(old_strings, new_strings)

    def _process_opcodes_safely(
        self, old_lines: List[str], new_lines: List[str], opcodes
    ) -> str:
        """Process opcodes with LaTeX safety checks"""
        result = []

        for opcode in opcodes:
            tag = opcode.tag
            i1, i2, j1, j2 = opcode.i1, opcode.i2, opcode.j1, opcode.j2

            if tag == "equal":
                result.extend(old_lines[i1:i2])

            elif tag == "delete":
                if not self.config.no_del:
                    deleted_content = old_lines[i1:i2]
                    for content in deleted_content:
                        if self._is_safe_for_markup(content):
                            result.append(f"\\DIFdel{{{content}}}")
                        else:
                            # Skip unsafe content or add it without markup
                            if self.config.debug:
                                print(
                                    f"Skipping unsafe delete markup: {content[:50]}...",
                                    file=sys.stderr,
                                )
                            result.append(content)

            elif tag == "insert":
                inserted_content = new_lines[j1:j2]
                for content in inserted_content:
                    if self._is_safe_for_markup(content):
                        result.append(f"\\DIFadd{{{content}}}")
                    else:
                        # Skip unsafe content or add it without markup
                        if self.config.debug:
                            print(
                                f"Skipping unsafe insert markup: {content[:50]}...",
                                file=sys.stderr,
                            )
                        result.append(content)

            elif tag == "replace":
                # Handle replacements carefully
                old_content = old_lines[i1:i2]
                new_content = new_lines[j1:j2]

                # Check if we can safely markup the replacement
                old_safe = all(self._is_safe_for_markup(c) for c in old_content)
                new_safe = all(self._is_safe_for_markup(c) for c in new_content)

                if old_safe and new_safe:
                    if not self.config.no_del:
                        for content in old_content:
                            result.append(f"\\DIFdel{{{content}}}")
                    for content in new_content:
                        result.append(f"\\DIFadd{{{content}}}")
                else:
                    # Fallback: just use new content without markup
                    result.extend(new_content)

            elif tag == "move":
                if not self.config.no_del:
                    moved_content = old_lines[i1:i2]
                    for content in moved_content:
                        if self._is_safe_for_markup(content):
                            result.append(f"\\DIFmovefrom{{{content}}}")
                        else:
                            result.append(content)

            elif tag == "moved":
                moved_content = new_lines[j1:j2]
                for content in moved_content:
                    if self._is_safe_for_markup(content):
                        result.append(f"\\DIFmoveto{{{content}}}")
                    else:
                        result.append(content)

        return "".join(result)

    def _is_safe_for_markup(self, content: str) -> bool:
        """Check if content is safe to wrap in diff markup"""
        content = content.strip()

        # Empty content is not safe
        if not content:
            return False

        # Don't markup content that starts with LaTeX environments
        if re.match(r"\\begin\{", content) or re.match(r"\\end\{", content):
            return False

        # Don't markup math delimiters
        if content.startswith("$") or content.endswith("$"):
            return False

        # Don't markup if it contains unbalanced braces
        if not self._has_balanced_braces(content):
            return False

        # Don't markup incomplete commands
        if content.endswith("\\") or re.search(r"\\[a-zA-Z]+$", content):
            return False

        # Don't markup isolated labels or equation numbers
        if re.match(r"\\label\{[^}]*\}$", content.strip()):
            return False

        return True

    def _has_balanced_braces(self, text: str) -> bool:
        """Check if braces are balanced in the text"""
        depth = 0
        escaped = False

        for char in text:
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    return False

        return depth == 0

    def _simple_diff(self, old_strings: List[str], new_strings: List[str]) -> str:
        """Simple fallback diff implementation"""
        result = []

        # Very basic approach: find common elements
        old_set = set(old_strings)
        new_set = set(new_strings)
        common = old_set.intersection(new_set)

        # Process old strings
        for item in old_strings:
            if item in common:
                result.append(item)
            elif not self.config.no_del and self._is_safe_for_markup(item):
                result.append(f"\\DIFdel{{{item}}}")

        # Add new strings that aren't common
        for item in new_strings:
            if item not in common and self._is_safe_for_markup(item):
                result.append(f"\\DIFadd{{{item}}}")

        return "".join(result)

    def _combine_diff_document(
        self, preamble: str, body: str, old_file: str, new_file: str
    ) -> str:
        """Combine preamble and body into final diff document"""

        # Add header comments
        old_time = time.ctime(os.path.getmtime(old_file))
        new_time = time.ctime(os.path.getmtime(new_file))

        header = [
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
            "%DIF LATEXDIFF DIFFERENCE FILE",
            f"%DIF DEL {old_file}   {old_time}",
            f"%DIF ADD {new_file}   {new_time}",
        ]

        result = "\n".join(header) + "\n" + preamble + body

        return result


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Python implementation of latexdiff with enhanced move detection",
        epilog="This is a Python port of the original Perl latexdiff script.",
    )

    parser.add_argument("old_file", help="Old LaTeX file")
    parser.add_argument("new_file", help="New LaTeX file")

    parser.add_argument(
        "-t",
        "--type",
        default="UNDERLINE",
        choices=[
            "UNDERLINE",
            "CTRADITIONAL",
            "TRADITIONAL",
            "CFONT",
            "FONTSTRIKE",
            "INVISIBLE",
        ],
        help="Type of diff markup (default: UNDERLINE)",
    )

    parser.add_argument(
        "-s",
        "--subtype",
        default="SAFE",
        choices=["SAFE", "WHOLE", "LABEL"],
        help="Subtype of diff markup (default: SAFE)",
    )

    parser.add_argument(
        "-f",
        "--floattype",
        default="FLOATSAFE",
        choices=["FLOATSAFE", "TRADITIONAL", "IDENTICAL"],
        help="Float type handling (default: FLOATSAFE)",
    )

    parser.add_argument(
        "--math-markup",
        default="FINE",
        choices=["OFF", "WHOLE", "COARSE", "FINE"],
        help="Math markup mode (default: FINE)",
    )

    parser.add_argument(
        "--graphics-markup",
        default="NEWONLY",
        choices=["OFF", "NONE", "NEWONLY", "BOTH"],
        help="Graphics markup mode (default: NEWONLY)",
    )

    parser.add_argument(
        "-e", "--encoding", default="utf8", help="File encoding (default: utf8)"
    )

    parser.add_argument("-V", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument("--debug", action="store_true", help="Debug output")

    parser.add_argument("--no-del", action="store_true", help="Remove all deleted text")

    parser.add_argument(
        "--allow-spaces", action="store_true", help="Allow spaces in commands"
    )

    parser.add_argument("--flatten", action="store_true", help="Flatten included files")

    parser.add_argument(
        "-p", "--preamble", dest="preamble_file", help="Preamble file to include"
    )

    parser.add_argument(
        "--packages", action="append", default=[], help="Packages to consider"
    )

    parser.add_argument(
        "-L",
        "--label",
        dest="labels",
        action="append",
        default=[],
        help="Labels for old and new files",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"latexdiff.py {__version__} (based on latexdiff {__original_version__})",
    )

    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Check input files exist
    if not os.path.exists(args.old_file):
        print(f"Error: Input file {args.old_file} does not exist", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.new_file):
        print(f"Error: Input file {args.new_file} does not exist", file=sys.stderr)
        sys.exit(1)

    # Create configuration
    config = Config(
        type=args.type,
        subtype=args.subtype,
        floattype=args.floattype,
        math_markup=args.math_markup,
        graphics_markup=args.graphics_markup,
        encoding=args.encoding,
        verbose=args.verbose,
        debug=args.debug,
        no_del=args.no_del,
        allow_spaces=args.allow_spaces,
        flatten=args.flatten,
        preamble_file=args.preamble_file,
        packages=args.packages,
        labels=args.labels,
    )

    if config.verbose:
        print(
            f"latexdiff.py {__version__} (based on latexdiff {__original_version__})",
            file=sys.stderr,
        )
        print(f"Encoding: {config.encoding}", file=sys.stderr)

    # Create and run diff processor
    diff_processor = LaTeXDiff(config)

    try:
        result = diff_processor.process_files(args.old_file, args.new_file)
        print(result)
    except Exception as e:
        if config.debug:
            import traceback

            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
