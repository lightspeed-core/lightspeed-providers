"""
Unit tests for Solr filter conversion functions.

Tests the filter-to-Solr query conversion logic for both ComparisonFilter
and CompoundFilter types.
"""

import pytest
from llama_stack_api import ComparisonFilter, CompoundFilter

# pylint: disable=line-too-long
from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.solr import (
    _build_solr_filter_query,
    _filter_to_solr_fq,
)

# pylint: enable=line-too-long


class TestComparisonFilters:
    """Test conversion of ComparisonFilter to Solr fq syntax."""

    def test_eq_filter_with_string(self) -> None:
        """Test equality filter with string value."""
        filter_obj = ComparisonFilter(type="eq", key="platform", value="openshift")
        result = _filter_to_solr_fq(filter_obj)
        assert result == 'platform:"openshift"'

    def test_eq_filter_with_number(self) -> None:
        """Test equality filter with numeric value."""
        filter_obj = ComparisonFilter(type="eq", key="version", value=2.9)
        result = _filter_to_solr_fq(filter_obj)
        assert result == "version:2.9"

    def test_ne_filter(self) -> None:
        """Test not-equal filter."""
        filter_obj = ComparisonFilter(type="ne", key="status", value="archived")
        result = _filter_to_solr_fq(filter_obj)
        assert result == '-status:"archived"'

    def test_in_filter_with_strings(self) -> None:
        """Test 'in' filter with string values."""
        filter_obj = ComparisonFilter(
            type="in", key="platform", value=["openshift", "kubernetes", "ansible"]
        )
        result = _filter_to_solr_fq(filter_obj)
        assert result == 'platform:("openshift" OR "kubernetes" OR "ansible")'

    def test_in_filter_with_numbers(self) -> None:
        """Test 'in' filter with numeric values."""
        filter_obj = ComparisonFilter(type="in", key="version", value=[1, 2, 3])
        result = _filter_to_solr_fq(filter_obj)
        assert result == "version:(1 OR 2 OR 3)"

    def test_in_filter_empty_list(self) -> None:
        """Test 'in' filter with empty list matches nothing."""
        filter_obj = ComparisonFilter(type="in", key="platform", value=[])
        result = _filter_to_solr_fq(filter_obj)
        assert result == "*:* NOT *:*"

    def test_nin_filter(self) -> None:
        """Test 'not in' filter."""
        filter_obj = ComparisonFilter(
            type="nin", key="status", value=["draft", "deleted"]
        )
        result = _filter_to_solr_fq(filter_obj)
        assert result == '-status:("draft" OR "deleted")'

    def test_nin_filter_empty_list(self) -> None:
        """Test 'not in' filter with empty list matches everything."""
        filter_obj = ComparisonFilter(type="nin", key="status", value=[])
        result = _filter_to_solr_fq(filter_obj)
        assert result == "*:*"

    def test_string_escaping(self) -> None:
        """Test special character escaping in string values."""
        filter_obj = ComparisonFilter(type="eq", key="title", value='Test "quote"')
        result = _filter_to_solr_fq(filter_obj)
        assert result == 'title:"Test \\"quote\\""'

    def test_in_filter_with_non_list_raises_error(self) -> None:
        """Test that 'in' filter with non-list value raises ValueError."""
        filter_obj = ComparisonFilter(type="in", key="platform", value="openshift")
        with pytest.raises(ValueError, match="'in' filter requires a list value"):
            _filter_to_solr_fq(filter_obj)

    def test_invalid_field_name_raises_error(self) -> None:
        """Test that invalid field names are rejected."""
        # Field names with spaces are invalid
        filter_obj = ComparisonFilter(type="eq", key="invalid field", value="test")
        with pytest.raises(ValueError, match="Invalid field name"):
            _filter_to_solr_fq(filter_obj)

    def test_field_name_with_special_chars_raises_error(self) -> None:
        """Test that field names with special characters are rejected."""
        # Field names with special characters like $ are invalid
        filter_obj = ComparisonFilter(type="eq", key="field$name", value="test")
        with pytest.raises(ValueError, match="Invalid field name"):
            _filter_to_solr_fq(filter_obj)

    def test_valid_field_names_accepted(self) -> None:
        """Test that valid field names with dots, hyphens, and underscores work."""
        # Test dot notation
        filter_obj = ComparisonFilter(type="eq", key="metadata.platform", value="test")
        result = _filter_to_solr_fq(filter_obj)
        assert result == 'metadata.platform:"test"'

        # Test hyphen
        filter_obj = ComparisonFilter(type="eq", key="my-field", value="test")
        result = _filter_to_solr_fq(filter_obj)
        assert result == 'my-field:"test"'

        # Test underscore
        filter_obj = ComparisonFilter(type="eq", key="_private_field", value="test")
        result = _filter_to_solr_fq(filter_obj)
        assert result == '_private_field:"test"'

    def test_unsupported_range_operators_raise_error(self) -> None:
        """Test that range operators (gt, gte, lt, lte) are not supported."""
        # Test gt
        filter_obj = ComparisonFilter(type="gt", key="score", value=0.8)
        with pytest.raises(ValueError, match="Solr only supports: eq, ne, in, nin"):
            _filter_to_solr_fq(filter_obj)

        # Test gte
        filter_obj = ComparisonFilter(type="gte", key="score", value=0.5)
        with pytest.raises(ValueError, match="Solr only supports: eq, ne, in, nin"):
            _filter_to_solr_fq(filter_obj)

        # Test lt
        filter_obj = ComparisonFilter(type="lt", key="rank", value=10)
        with pytest.raises(ValueError, match="Solr only supports: eq, ne, in, nin"):
            _filter_to_solr_fq(filter_obj)

        # Test lte
        filter_obj = ComparisonFilter(type="lte", key="rank", value=5)
        with pytest.raises(ValueError, match="Solr only supports: eq, ne, in, nin"):
            _filter_to_solr_fq(filter_obj)


class TestCompoundFilters:
    """Test conversion of CompoundFilter to Solr fq syntax."""

    def test_and_filter(self) -> None:
        """Test AND compound filter."""
        filter_obj = CompoundFilter(
            type="and",
            filters=[
                ComparisonFilter(type="eq", key="platform", value="openshift"),
                ComparisonFilter(type="eq", key="version", value="4.12"),
            ],
        )
        result = _filter_to_solr_fq(filter_obj)
        assert result == '(platform:"openshift" AND version:"4.12")'

    def test_or_filter(self) -> None:
        """Test OR compound filter."""
        filter_obj = CompoundFilter(
            type="or",
            filters=[
                ComparisonFilter(type="eq", key="platform", value="openshift"),
                ComparisonFilter(type="eq", key="platform", value="kubernetes"),
            ],
        )
        result = _filter_to_solr_fq(filter_obj)
        assert result == '(platform:"openshift" OR platform:"kubernetes")'

    def test_nested_compound_filter(self) -> None:
        """Test nested compound filters."""
        filter_obj = CompoundFilter(
            type="and",
            filters=[
                ComparisonFilter(type="eq", key="status", value="published"),
                CompoundFilter(
                    type="or",
                    filters=[
                        ComparisonFilter(type="eq", key="platform", value="openshift"),
                        ComparisonFilter(type="eq", key="platform", value="ansible"),
                    ],
                ),
            ],
        )
        result = _filter_to_solr_fq(filter_obj)
        expected = (
            '(status:"published" AND ' + '(platform:"openshift" OR platform:"ansible"))'
        )
        assert result == expected

    def test_empty_compound_filter(self) -> None:
        """Test empty compound filter matches all."""
        filter_obj = CompoundFilter(type="and", filters=[])
        result = _filter_to_solr_fq(filter_obj)
        assert result == "*:*"


class TestBuildSolrFilterQuery:
    """Test combining static chunk filter with dynamic filters."""

    def test_no_filters(self) -> None:
        """Test with no filters returns None."""
        result = _build_solr_filter_query(None, None)
        assert result is None

    def test_only_static_filter(self) -> None:
        """Test with only static filter."""
        result = _build_solr_filter_query("is_chunk:true", None)
        assert result == "is_chunk:true"

    def test_only_dynamic_filter(self) -> None:
        """Test with only dynamic filter."""
        filter_obj = ComparisonFilter(type="eq", key="platform", value="openshift")
        result = _build_solr_filter_query(None, filter_obj)
        assert result == 'platform:"openshift"'

    def test_combined_static_and_dynamic_filters(self) -> None:
        """Test combining static and dynamic filters with AND."""
        filter_obj = ComparisonFilter(type="eq", key="platform", value="openshift")
        result = _build_solr_filter_query("is_chunk:true", filter_obj)
        assert result == '(is_chunk:true AND platform:"openshift")'

    def test_combined_with_compound_filter(self) -> None:
        """Test combining static filter with compound dynamic filter."""
        filter_obj = CompoundFilter(
            type="and",
            filters=[
                ComparisonFilter(type="eq", key="platform", value="openshift"),
                ComparisonFilter(type="ne", key="status", value="archived"),
            ],
        )
        result = _build_solr_filter_query("is_chunk:true", filter_obj)
        expected = '(is_chunk:true AND (platform:"openshift" AND -status:"archived"))'
        assert result == expected
