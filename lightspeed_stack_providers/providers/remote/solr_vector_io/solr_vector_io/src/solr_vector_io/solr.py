"""Backward-compatible exports for the Solr vector IO provider."""

from .adapter import SolrVectorIOAdapter
from .constants import VECTOR_DBS_PREFIX, VERSION
from .solr_index import OKP_SOURCE, SolrIndex

__all__ = [
    "OKP_SOURCE",
    "SolrIndex",
    "SolrVectorIOAdapter",
    "VECTOR_DBS_PREFIX",
    "VERSION",
]
