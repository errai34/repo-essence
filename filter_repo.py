#!/usr/bin/env python3
"""
Collect tutorial/example/demo files (in full) and their local Python dependencies
from a merged XML representation of a repository. Output a single compressed XML
that replicates the original structure (<file_summary>, <repository_structure>,
<repository_files>) while staying under a specified token limit using tiktoken.

Usage:
  python gather_tutorials.py --merged-xml path/to/big.xml \
                             --config path/to/filter_config.xml \
                             --output out.xml \
                             --max-tokens 30000 \
                             --model gpt-3.5-turbo

Dependencies:
  pip install tiktoken
"""

import argparse
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
import os

# For GPT-based token counting
import tiktoken

################################################################################
# FILTER RULES (similar to your existing approach)
################################################################################


class FilterRule(ABC):
    @abstractmethod
    def matches(self, path: str, content: str) -> bool:
        pass


class PathPatternRule(FilterRule):
    def __init__(self, pattern: str, case_sensitive: bool = False):
        flags = 0 if not case_sensitive else 0
        self.pattern = re.compile(pattern, flags)

    def matches(self, path: str, content: str) -> bool:
        return bool(self.pattern.search(path))


class ContentPatternRule(FilterRule):
    def __init__(self, pattern: str, case_sensitive: bool = False):
        flags = 0 if not case_sensitive else 0
        self.pattern = re.compile(pattern, flags)

    def matches(self, path: str, content: str) -> bool:
        return bool(self.pattern.search(content))


class AndRule(FilterRule):
    def __init__(self, rules: List[FilterRule]):
        self.rules = rules

    def matches(self, path: str, content: str) -> bool:
        return all(r.matches(path, content) for r in self.rules)


class OrRule(FilterRule):
    def __init__(self, rules: List[FilterRule]):
        self.rules = rules

    def matches(self, path: str, content: str) -> bool:
        return any(r.matches(path, content) for r in self.rules)


class NotRule(FilterRule):
    def __init__(self, rule: FilterRule):
        self.rule = rule

    def matches(self, path: str, content: str) -> bool:
        return not self.rule.matches(path, content)


################################################################################
# RULE FACTORY
################################################################################


class RuleFactory:
    @staticmethod
    def create_from_xml(elem: ET.Element) -> FilterRule:
        tag = elem.tag.lower()
        if tag == "path_pattern":
            return PathPatternRule(
                elem.get("pattern", ""),
                case_sensitive=(elem.get("case_sensitive", "false").lower() == "true"),
            )
        elif tag == "content_pattern":
            return ContentPatternRule(
                elem.get("pattern", ""),
                case_sensitive=(elem.get("case_sensitive", "false").lower() == "true"),
            )
        elif tag == "and":
            return AndRule([RuleFactory.create_from_xml(c) for c in elem])
        elif tag == "or":
            return OrRule([RuleFactory.create_from_xml(c) for c in elem])
        elif tag == "not":
            if len(elem) != 1:
                raise ValueError("'not' rule must have exactly one child.")
            return NotRule(RuleFactory.create_from_xml(elem[0]))
        else:
            raise ValueError(f"Unknown rule type: {tag}")


################################################################################
# TOKEN COUNTING WITH TIKTOKEN
################################################################################


def load_token_encoder(model_name: str = "gpt-3.5-turbo"):
    """
    Returns a tiktoken encoding object for the specified model.
    Adjust if you use a different model (like gpt-4, etc.).
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # fallback if model not recognized
        enc = tiktoken.get_encoding("cl100k_base")
    return enc


def count_tokens(text: str, encoder) -> int:
    """
    Count tokens using tiktoken for a given text input.
    """
    return len(encoder.encode(text))


################################################################################
# PARSE THE BIG XML FILE
################################################################################


def extract_tag_content(src: str, tagname: str) -> Optional[str]:
    """
    Extract <tagname> ... </tagname> from src using a simple regex.
    Return the content including the tag itself, or None if not found.
    """
    pattern = rf"(<{tagname}[\s>].*?</{tagname}>)"
    match = re.search(pattern, src, flags=re.DOTALL)
    if match:
        return match.group(1)
    return None


def remove_tag_from_text(src: str, tagname: str) -> str:
    """
    Remove the <tagname>...</tagname> block entirely from src.
    """
    pattern = rf"<{tagname}[\s>].*?</{tagname}>"
    return re.sub(pattern, "", src, flags=re.DOTALL)


def parse_files_section(src: str) -> List[ET.Element]:
    """
    Extract all <file path="...">...</file> blocks as Element objects.
    """
    file_pattern = re.compile(r'<file\s+path="([^"]+)">(.*?)</file>', re.DOTALL)
    matches = file_pattern.findall(src)
    file_elems = []
    for path, content in matches:
        f = ET.Element("file")
        f.set("path", path)
        f.text = content
        file_elems.append(f)
    return file_elems


################################################################################
# DISCOVER LOCAL PYTHON DEPENDENCIES
################################################################################

# Naive import pattern: either "import foo.bar" or "from foo.bar import something"
IMPORT_REGEX = re.compile(
    r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+.*|import\s+([A-Za-z0-9_\.]+))",
    re.MULTILINE,
)


def discover_local_imports(file_content: str) -> Set[str]:
    """
    Return a set of top-level module paths mentioned in the file's import statements.
    E.g. "from gala.coordinates import something" => "gala.coordinates"
         "import gala.dynamics" => "gala.dynamics"
    """
    found = set()
    for match in IMPORT_REGEX.findall(file_content):
        # each match is a tuple (group1, group2)
        # only one is non-empty
        mod = match[0] if match[0] else match[1]
        if mod:
            found.add(mod.strip())
    return found


def build_python_module_index(file_elems: List[ET.Element]) -> Dict[str, ET.Element]:
    """
    Build a naive mapping "package.subpackage.module" -> <file> element
    from all .py files. Also handle __init__.py as "package.subpackage".
    """
    index = {}
    for fe in file_elems:
        path = fe.get("path", "")
        if not path.endswith(".py"):
            continue
        # convert path like "gala/coordinates/greatcircle.py" -> "gala.coordinates.greatcircle"
        parts = path.split("/")
        dotted_parts = []
        for p in parts:
            if p.endswith(".py"):
                dotted_parts.append(p[:-3])  # drop .py
            else:
                dotted_parts.append(p)
        if dotted_parts[-1] == "__init__":
            dotted_parts.pop()  # so "gala/coordinates/__init__.py" => "gala.coordinates"
        dotted_name = ".".join(dotted_parts)
        # store the <file> element
        index[dotted_name] = fe
    return index


################################################################################
# BFS: Gather tutorial/demo files + their dependencies
################################################################################


def gather_tutorials_and_dependencies(
    tutorial_rule: FilterRule, file_elems: List[ET.Element], encoder, max_tokens: int
) -> List[ET.Element]:
    """
    1) Identify all tutorial-like files (matching the rule).
    2) BFS to include them + any local python dependencies.
    3) Skip large data files or anything that would exceed token limit.
    """
    # Convert the list into a path -> element map for quick lookup
    path_map: Dict[str, ET.Element] = {f.get("path", ""): f for f in file_elems}

    # Build a naive Python module index
    module_index = build_python_module_index(file_elems)

    # Start BFS queue
    queue: List[ET.Element] = []
    included_paths = set()
    total_tokens = 0
    included_files: List[ET.Element] = []

    # Step A: find all tutorial files
    for fe in file_elems:
        p = fe.get("path", "")
        c = fe.text or ""
        if tutorial_rule.matches(p, c):
            queue.append(fe)

    # BFS
    while queue:
        fe = queue.pop(0)
        path = fe.get("path", "")
        content = fe.text or ""
        if path in included_paths:
            continue

        # Skip data or large binaries
        if re.search(r"\.(csv|dat|txt|json|md5|h5|pot|coeff)$", path, re.IGNORECASE):
            continue

        # Count tokens in this file
        t_count = count_tokens(content, encoder)
        if total_tokens + t_count > max_tokens:
            # can't include this file without exceeding budget
            continue

        # Ok we include it
        included_paths.add(path)
        included_files.append(fe)
        total_tokens += t_count

        # If it's a .py, discover local imports
        if path.endswith(".py"):
            local_imports = discover_local_imports(content)
            for mod_name in local_imports:
                # if we find that mod_name in module_index, queue it
                if mod_name in module_index:
                    dep_elem = module_index[mod_name]
                    dep_path = dep_elem.get("path", "")
                    if dep_path not in included_paths:
                        queue.append(dep_elem)

    return included_files


################################################################################
# MAIN
################################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Gather tutorials + code deps into a single compressed XML under a token limit."
    )
    parser.add_argument(
        "--merged-xml", required=True, help="Path to the big merged .xml file"
    )
    parser.add_argument(
        "--config", required=True, help="Filter config XML (for tutorial-like rules)"
    )
    parser.add_argument(
        "--output", help="Path for the output file (optional, will auto-construct if not provided)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30000,
        help="Maximum number of tokens to include (default: 30000)",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Model to use for token counting (default: gpt-3.5-turbo)",
    )
    args = parser.parse_args()

    # Auto-construct output filename if not provided
    if not args.output:
        # Extract the base name from input XML
        input_base = os.path.basename(args.merged_xml)
        input_name = os.path.splitext(input_base)[0]
        
        # Create filtered_repos directory if it doesn't exist
        os.makedirs("filtered_repos", exist_ok=True)
        
        # Construct output filename: filtered_repos/name.filtered-{tokens}k.xml
        token_k = args.max_tokens // 1000
        args.output = f"filtered_repos/{input_name}.filtered-{token_k}k.xml"

    # Load the tiktoken encoder
    encoder = load_token_encoder(args.model)

    # 1. Read the entire merged XML as raw text
    with open(args.merged_xml, "r", encoding="utf-8") as f:
        merged_text = f.read()

    # 2. Extract <file_summary> block, <repository_structure> block
    file_summary_block = extract_tag_content(merged_text, "file_summary")
    repo_structure_block = extract_tag_content(merged_text, "repository_structure")

    # 3. Remove them from the text so we can parse <file> blocks cleanly
    text_without_headers = merged_text
    if file_summary_block:
        text_without_headers = remove_tag_from_text(
            text_without_headers, "file_summary"
        )
    if repo_structure_block:
        text_without_headers = remove_tag_from_text(
            text_without_headers, "repository_structure"
        )

    # 4. Parse <file path="..."> blocks from the remainder
    all_file_elems = parse_files_section(text_without_headers)

    # 5. Load the filter config (to find tutorial/demo files)
    config_tree = ET.parse(args.config)
    config_root = config_tree.getroot()
    tutorial_rule = RuleFactory.create_from_xml(config_root)

    # 6. BFS gather tutorials + deps
    included_files = gather_tutorials_and_dependencies(
        tutorial_rule=tutorial_rule,
        file_elems=all_file_elems,
        encoder=encoder,
        max_tokens=args.max_tokens,
    )

    # 7. Build final XML
    #    We'll replicate a "compressed_repository" top-level, then embed
    #    <file_summary>, <repository_structure>, and <repository_files>.

    root = ET.Element("compressed_repository")

    # Insert <file_summary> if we have it
    if file_summary_block:
        # parse as an Element so we preserve valid XML
        fs_elem = ET.fromstring(file_summary_block)
        root.append(fs_elem)

    # Insert <repository_structure> if we have it
    if repo_structure_block:
        rs_elem = ET.fromstring(repo_structure_block)
        root.append(rs_elem)

    # Create <repository_files> in the output
    repo_files_elem = ET.SubElement(root, "repository_files")

    # Insert included <file> elements
    total_tokens = 0
    for fe in included_files:
        path = fe.get("path", "")
        content = fe.text or ""
        # we can count tokens again or re-use BFS totals
        t_count = count_tokens(content, encoder)
        total_tokens += t_count

        # Make a new <file> with the same structure
        new_file = ET.SubElement(repo_files_elem, "file")
        new_file.set("path", path)
        new_file.text = content

    # Optionally record total token usage
    root.set("total_tokens", str(total_tokens))

    # 8. Pretty-print
    xml_str = ET.tostring(root, encoding="unicode")
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # 9. Write to output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"Done! Wrote {len(included_files)} files to {args.output}.")
    print(f"Total tokens used: {total_tokens} (model: {args.model})")


if __name__ == "__main__":
    main()
