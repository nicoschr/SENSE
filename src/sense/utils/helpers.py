"""
Common helper functions used throughout SENSE.
"""

from __future__ import annotations

import re
from typing import Any, Optional


def norm_code(x: Any) -> Optional[str]:
    """
    Normalize activity codes to simple stripped strings.

    Handles various formats that may occur in Brightway databases:
    - ('abc',) -> "abc"
    - ["abc"]  -> "abc"
    - None     -> None
    - everything else -> str(x).strip()
    
    Args:
        x: The code to normalize (may be a string, tuple, list, or None)
        
    Returns:
        Normalized string code, or None if input was None
        
    Examples:
        >>> norm_code("abc")
        'abc'
        >>> norm_code(('abc',))
        'abc'
        >>> norm_code(['abc'])
        'abc'
        >>> norm_code(None) is None
        True
    """
    if x is None:
        return None
    if isinstance(x, (tuple, list)) and len(x) == 1:
        return str(x[0]).strip()
    return str(x).strip()


def safe_name(name: str) -> str:
    """
    Normalize strings for use in filenames.
    
    Replaces problematic characters with underscores and collapses
    multiple underscores into one.
    
    Args:
        name: The string to normalize
        
    Returns:
        Filesystem-safe string
        
    Examples:
        >>> safe_name("my/file:name")
        'my_file_name'
        >>> safe_name("hello  world")
        'hello_world'
    """
    s = str(name)
    bad = "/\\:- "
    for ch in bad:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def slug(s: str) -> str:
    """
    Turn an arbitrary string into a filesystem-safe slug.
    
    Converts to lowercase, replaces whitespace with underscores,
    and removes non-alphanumeric characters.
    
    Args:
        s: Input string
        
    Returns:
        Slugified string
        
    Examples:
        >>> slug("ecoinvent-3.10-cutoff regionalized")
        'ecoinvent-3-10-cutoff_regionalized'
        >>> slug("Hello World!")
        'hello_world'
    """
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "-", s)
    return s.strip("-_")
