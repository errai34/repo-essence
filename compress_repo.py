#!/usr/bin/env python3
"""
Filter and process XML repository representation based on configurable rules.
Supports flexible filtering criteria and rule combinations.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import xml.dom.minidom

# -------------------------------------------------------------------
# FILTER RULES
# -------------------------------------------------------------------
class FilterRule(ABC):
    """Base class for all filter rules"""
    @abstractmethod
    def matches(self, path: str, content: str) -> bool:
        """Return True if the file matches this rule"""
        pass

class PathPatternRule(FilterRule):
    """Filter based on file path pattern"""
    def __init__(self, pattern: str, case_sensitive: bool = False):
        flags = 0 if case_sensitive else re.IGNORECASE
        self.pattern = re.compile(pattern, flags)
    
    def matches(self, path: str, content: str) -> bool:
        return bool(self.pattern.search(path))

class ContentPatternRule(FilterRule):
    """Filter based on file content pattern"""
    def __init__(self, pattern: str, case_sensitive: bool = False):
        flags = 0 if case_sensitive else re.IGNORECASE
        self.pattern = re.compile(pattern, flags)
    
    def matches(self, path: str, content: str) -> bool:
        return bool(self.pattern.search(content))

class AndRule(FilterRule):
    """Combines multiple rules with AND logic"""
    def __init__(self, rules: List[FilterRule]):
        self.rules = rules
    
    def matches(self, path: str, content: str) -> bool:
        return all(rule.matches(path, content) for rule in self.rules)

class OrRule(FilterRule):
    """Combines multiple rules with OR logic"""
    def __init__(self, rules: List[FilterRule]):
        self.rules = rules
    
    def matches(self, path: str, content: str) -> bool:
        return any(rule.matches(path, content) for rule in self.rules)

class NotRule(FilterRule):
    """Negates another rule"""
    def __init__(self, rule: FilterRule):
        self.rule = rule
    
    def matches(self, path: str, content: str) -> bool:
        return not self.rule.matches(path, content)

# -------------------------------------------------------------------
# RULE FACTORY
# -------------------------------------------------------------------
class RuleFactory:
    """Creates filter rules from configuration"""
    @staticmethod
    def create_from_xml(elem: ET.Element) -> FilterRule:
        rule_type = elem.tag
        
        if rule_type == "path_pattern":
            return PathPatternRule(
                elem.get("pattern"),
                elem.get("case_sensitive", "false").lower() == "true"
            )
        elif rule_type == "content_pattern":
            return ContentPatternRule(
                elem.get("pattern"),
                elem.get("case_sensitive", "false").lower() == "true"
            )
        elif rule_type == "and":
            rules = [RuleFactory.create_from_xml(child) for child in elem]
            return AndRule(rules)
        elif rule_type == "or":
            rules = [RuleFactory.create_from_xml(child) for child in elem]
            return OrRule(rules)
        elif rule_type == "not":
            if len(elem) != 1:
                raise ValueError("'not' rule must have exactly one child rule")
            rule = RuleFactory.create_from_xml(elem[0])
            return NotRule(rule)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

# -------------------------------------------------------------------
# XML PROCESSING
# -------------------------------------------------------------------
def parse_merged_xml(xml_path: str) -> ET.Element:
    """Parse the input XML file"""
    with open(xml_path) as f:
        content = f.read()
    
    # Find all file entries using regex
    file_matches = re.finditer(r'<file path="([^"]+)">(.*?)</file>', content, re.DOTALL)
    
    # Create new XML structure
    root = ET.Element("root")
    repo_files = ET.SubElement(root, "repository_files")
    
    for match in file_matches:
        path = match.group(1)
        content = match.group(2)
        
        file_elem = ET.SubElement(repo_files, "file")
        file_elem.set("path", path)
        file_elem.text = content
    
    return root

def count_tokens(text: str) -> int:
    """Rough estimate of token count based on whitespace and punctuation"""
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return len(tokens)

def score_file_content(path: str, content: str) -> float:
    """Score file content based on its likely importance"""
    import re
    
    # Files we want to exclude or deprioritize
    if re.search(r'[/\\](?:tests?|data)[/\\].*\.(?:txt|dat|csv)$', path, re.I):
        return 0.0
        
    score = 1.0
    
    # Prioritize certain file types
    if path.endswith('.py'):
        score *= 2.0
    elif path.endswith(('.rst', '.md')):
        score *= 1.5
    elif path.endswith('.yml'):
        score *= 1.2
        
    # Boost score based on content indicators
    if re.search(r'class\s+\w+\([^)]*\):', content):
        score *= 1.5
    if re.search(r'def\s+\w+\([^)]*\):', content):
        score *= 1.3
    if '"""' in content:
        score *= 1.2
    
    # Penalize files that are mostly data
    if re.search(r'^[\d\s.,+-]+$', content.strip(), re.M):
        score *= 0.1
        
    return score

def filter_xml_with_token_threshold(root: ET.Element, rule: FilterRule, min_tokens: int = 95000, max_tokens: int = 100000) -> ET.Element:
    """Filter XML while ensuring token count is between min and max"""
    # First pass: collect all matching files
    matching_files = []
    for file_elem in root.findall('.//file'):
        path = file_elem.get('path', '')
        content = ''.join(file_elem.itertext())
        tokens = count_tokens(content)
        if rule.matches(path, content):
            score = score_file_content(path, content)
            if score > 0:  # Only include files with positive scores
                matching_files.append((file_elem, tokens, score))
    
    # Sort by score-to-token ratio (descending) to prioritize high-value content
    matching_files.sort(key=lambda x: x[2]/x[1], reverse=True)
    
    # Create new root with repository_files
    new_root = ET.Element('root')
    repo_files = ET.SubElement(new_root, 'repository_files')
    total_tokens = 0
    
    print("\nSelected files:")
    print(f"{'File Path':<60} {'Tokens':>8} {'Score':>8} {'Value/Token':>12}")
    print("-" * 90)
    
    # Add files until we get close to max_tokens
    for file_elem, tokens, score in matching_files:
        if total_tokens + tokens <= max_tokens:
            path = file_elem.get('path', '')
            repo_files.append(file_elem)
            total_tokens += tokens
            print(f"{path:<60} {tokens:>8} {score:>8.2f} {score/tokens:>12.6f}")
            if total_tokens >= min_tokens:
                break
    
    print("-" * 90)
    print(f"Total tokens in filtered output: {total_tokens}")
    return new_root

# -------------------------------------------------------------------
# MAIN CLI
# -------------------------------------------------------------------
def main():
    import argparse
    from pathlib import Path
    import xml.dom.minidom
    import os

    ap = argparse.ArgumentParser(
        description="Filter repository XML based on configurable rules."
    )
    ap.add_argument(
        "merged_xml",
        help="Path to merged representation file (the big .xml)."
    )
    ap.add_argument(
        "--config",
        type=str,
        help="Path to filter configuration XML file",
        required=True
    )
    ap.add_argument(
        "--output",
        type=str,
        help="Path to output filtered XML file (default: input_name_filtered.xml)",
        default=None
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        help="Minimum number of tokens in output (default: 95000)",
        default=95000
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens in output (default: 100000)",
        default=100000
    )
    args = ap.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.merged_xml)
        args.output = os.path.join("repopack-filtered", f"{input_path.stem}_filtered{input_path.suffix}")
    
    # Load filter configuration
    config_tree = ET.parse(args.config)
    config_root = config_tree.getroot()
    
    # Create filter rule from config
    rule = RuleFactory.create_from_xml(config_root)
    
    # Parse and filter XML
    root = parse_merged_xml(args.merged_xml)
    filtered_root = filter_xml_with_token_threshold(root, rule, args.min_tokens, args.max_tokens)
    
    # Convert to string with pretty printing
    xml_str = ET.tostring(filtered_root, encoding='unicode')
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    # Write to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    print(f"Filtered XML saved to: {args.output}")

if __name__ == "__main__":
    main()
