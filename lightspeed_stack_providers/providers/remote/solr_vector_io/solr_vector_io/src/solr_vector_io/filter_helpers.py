"""Helper functions for converting filters to Solr filter query (fq) syntax."""

import re
from typing import Any, Optional

from llama_stack.providers.utils.vector_io.filters import (
    ComparisonFilter,
    CompoundFilter,
    Filter,
)

# Pattern for valid Solr field names: start with letter or underscore,
# followed by letters, digits, underscores, dots, or hyphens
_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def validate_field_name(field_name: str) -> None:
    """
    Validate that a field name is safe for use in Solr queries.

    Parameters:
        field_name: The field name to validate.

    Raises:
        ValueError: If the field name contains invalid characters.
    """
    if not _FIELD_NAME_PATTERN.match(field_name):
        raise ValueError(
            f"Invalid field name '{field_name}': must start with a letter or "
            f"underscore and contain only letters, digits, underscores, dots, or hyphens"
        )


def escape_solr_value(value: str) -> str:
    """
    Escape special characters in Solr query string values.

    Parameters:
        value: The string value to escape.

    Returns:
        Escaped string safe for use in Solr queries.
    """
    # Backslash must be escaped first to avoid double-escaping
    value = value.replace("\\", "\\\\")
    # Escape double quotes for quoted field values
    return value.replace('"', '\\"')


def format_solr_value(value: Any) -> str:
    """
    Format a value for use in Solr queries.

    Parameters:
        value: The value to format.

    Returns:
        Formatted value string.
    """
    if isinstance(value, str):
        escaped_value = escape_solr_value(value)
        return f'"{escaped_value}"'
    return str(value)


def handle_in_filter(key: str, value: Any, negate: bool = False) -> str:
    """
    Handle 'in' and 'nin' filters.

    Parameters:
        key: Field name.
        value: List of values.
        negate: Whether to negate the filter.

    Returns:
        Solr filter query string.

    Raises:
        ValueError: If value is not a list or field name is invalid.
    """
    # Validate field name for security
    validate_field_name(key)

    if not isinstance(value, list):
        op_name = "nin" if negate else "in"
        raise ValueError(f"'{op_name}' filter requires a list value, got {type(value)}")

    if not value:
        return "*:*" if negate else "*:* NOT *:*"

    or_parts = [format_solr_value(v) for v in value]
    filter_expr = f"{key}:({' OR '.join(or_parts)})"
    return f"-{filter_expr}" if negate else filter_expr


def _convert_comparison_filter(filter_obj: ComparisonFilter) -> str:
    """Convert a ComparisonFilter to Solr fq syntax.

    Parameters:
        filter_obj: ComparisonFilter object.

    Returns:
        Solr filter query string.

    Raises:
        ValueError: If filter type is unsupported.
    """
    key = filter_obj.key
    value = filter_obj.value
    filter_type = filter_obj.type

    # Validate field name for security
    validate_field_name(key)

    if filter_type == "eq":
        return f"{key}:{format_solr_value(value)}"
    if filter_type == "ne":
        return f"-{key}:{format_solr_value(value)}"
    if filter_type == "in":
        return handle_in_filter(key, value, negate=False)
    if filter_type == "nin":
        return handle_in_filter(key, value, negate=True)

    raise ValueError(
        f"Unsupported comparison filter type '{filter_type}'. "
        f"Solr only supports: eq, ne, in, nin"
    )


def _convert_compound_filter(filter_obj: CompoundFilter) -> str:
    """Convert a CompoundFilter to Solr fq syntax.

    Parameters:
        filter_obj: CompoundFilter object.

    Returns:
        Solr filter query string.

    Raises:
        ValueError: If filter type is unsupported.
    """
    filter_type = filter_obj.type
    sub_filters = filter_obj.filters

    if not sub_filters:
        return "*:*"

    # Recursively convert sub-filters
    converted_filters = [filter_to_solr_fq(f) for f in sub_filters]

    if filter_type == "and":
        return "(" + " AND ".join(converted_filters) + ")"
    if filter_type == "or":
        return "(" + " OR ".join(converted_filters) + ")"

    raise ValueError(f"Unsupported compound filter type: {filter_type}")


def filter_to_solr_fq(filter_obj: Filter) -> str:
    """
    Convert llama-stack Filter to Solr filter query (fq) syntax.

    Translates ComparisonFilter and CompoundFilter objects to Solr's
    native filter query language for metadata-based filtering.

    Parameters:
        filter_obj: Filter object (ComparisonFilter or CompoundFilter).

    Returns:
        Solr filter query string.

    Raises:
        ValueError: If filter type is unsupported or value type is invalid.
    """
    if isinstance(filter_obj, ComparisonFilter):
        return _convert_comparison_filter(filter_obj)

    if isinstance(filter_obj, CompoundFilter):
        return _convert_compound_filter(filter_obj)

    raise ValueError(f"Unsupported filter object type: {type(filter_obj)}")


def build_solr_filter_query(
    chunk_filter_query: Optional[str], filters: Optional[Filter]
) -> Optional[list[str] | str]:
    """
    Build Solr filter query combining static chunk filter and dynamic filters.

    Solr expects multiple fq parameters (implicitly ANDed) rather than a single
    combined boolean expression. Returns a list when multiple filters exist,
    or a single string for backward compatibility with single filters.

    Parameters:
        chunk_filter_query: Static filter query from config (e.g., "is_chunk:true").
        filters: Dynamic filters from the request.

    Returns:
        List of filter query strings (for multiple fq parameters),
        single filter string, or None if no filters.
    """
    filter_parts: list[str] = []

    # Add static chunk filter if configured
    if chunk_filter_query:
        filter_parts.append(chunk_filter_query)

    # Add dynamic filters if provided
    if filters:
        dynamic_fq = filter_to_solr_fq(filters)
        filter_parts.append(dynamic_fq)

    if not filter_parts:
        return None

    # Return list for multiple filters (Solr will AND them implicitly)
    # Return single string for backward compatibility
    if len(filter_parts) == 1:
        return filter_parts[0]

    return filter_parts
