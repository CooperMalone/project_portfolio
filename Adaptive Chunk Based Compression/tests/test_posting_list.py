"""
Comprehensive tests for the posting list module.
"""

import pytest
from utils.posting_list import (
    text_to_posting_list,
    documents_to_posting_list,
    merge_posting_lists,
    get_term_gaps,
)


class TestTextToPostingList:
    """Tests for text_to_posting_list function."""

    def test_single_word(self):
        """Test with a single word."""
        result = text_to_posting_list("hello", doc_id=1)
        assert result == {"hello": {1: [0]}}

    def test_multiple_words(self):
        """Test with multiple distinct words."""
        result = text_to_posting_list("hello world", doc_id=1)
        assert result == {
            "hello": {1: [0]},
            "world": {1: [1]},
        }

    def test_repeated_words(self):
        """Test with repeated words."""
        result = text_to_posting_list("the cat and the dog", doc_id=1)
        assert result == {
            "the": {1: [0, 3]},
            "cat": {1: [1]},
            "and": {1: [2]},
            "dog": {1: [4]},
        }

    def test_normalization_enabled(self):
        """Test that lowercase normalization works by default."""
        result = text_to_posting_list("Hello WORLD Hello", doc_id=1)
        assert result == {
            "hello": {1: [0, 2]},
            "world": {1: [1]},
        }

    def test_normalization_disabled(self):
        """Test with normalization disabled."""
        result = text_to_posting_list("Hello WORLD Hello", doc_id=1, normalize=False)
        assert result == {
            "Hello": {1: [0, 2]},
            "WORLD": {1: [1]},
        }

    def test_custom_doc_id(self):
        """Test with custom document ID."""
        result = text_to_posting_list("hello world", doc_id="doc_001")
        assert result == {
            "hello": {"doc_001": [0]},
            "world": {"doc_001": [1]},
        }

    def test_empty_string(self):
        """Test with empty string."""
        result = text_to_posting_list("", doc_id=1)
        assert result == {}

    def test_whitespace_handling(self):
        """Test with multiple spaces."""
        result = text_to_posting_list("hello   world", doc_id=1)
        assert result == {
            "hello": {1: [0]},
            "world": {1: [1]},
        }

    def test_complex_text(self):
        """Test with more complex text."""
        text = "the quick brown fox jumps over the lazy dog"
        result = text_to_posting_list(text, doc_id=1)
        assert result["the"] == {1: [0, 6]}
        assert result["fox"] == {1: [3]}
        assert result["jumps"] == {1: [4]}
        assert len(result) == 8  # 8 unique words


class TestDocumentsToPostingList:
    """Tests for documents_to_posting_list function."""

    def test_single_document(self):
        """Test with a single document."""
        result = documents_to_posting_list(["hello world"])
        assert result == {
            "hello": {0: [0]},
            "world": {0: [1]},
        }

    def test_multiple_documents(self):
        """Test with multiple documents."""
        docs = ["hello world", "hello there"]
        result = documents_to_posting_list(docs)
        assert result["hello"] == {0: [0], 1: [0]}
        assert result["world"] == {0: [1]}
        assert result["there"] == {1: [1]}

    def test_document_ids_assigned(self):
        """Test that documents are assigned sequential IDs."""
        docs = ["dog", "cat", "bird"]
        result = documents_to_posting_list(docs)
        assert "dog" in result and 0 in result["dog"]
        assert "cat" in result and 1 in result["cat"]
        assert "bird" in result and 2 in result["bird"]

    def test_shared_terms_across_docs(self):
        """Test terms appearing in multiple documents."""
        docs = ["apple banana", "banana orange", "apple orange"]
        result = documents_to_posting_list(docs)
        assert result["apple"] == {0: [0], 2: [0]}
        assert result["banana"] == {0: [1], 1: [0]}
        assert result["orange"] == {1: [1], 2: [1]}

    def test_empty_documents_list(self):
        """Test with empty documents list."""
        result = documents_to_posting_list([])
        assert result == {}

    def test_document_with_repeated_terms(self):
        """Test documents containing repeated terms."""
        docs = ["apple apple", "banana banana banana"]
        result = documents_to_posting_list(docs)
        assert result["apple"] == {0: [0, 1]}
        assert result["banana"] == {1: [0, 1, 2]}


class TestMergePostingLists:
    """Tests for merge_posting_lists function."""

    def test_merge_disjoint_posting_lists(self):
        """Test merging posting lists with no overlapping terms."""
        pl1 = {"hello": {1: [0]}}
        pl2 = {"world": {2: [0]}}
        result = merge_posting_lists([pl1, pl2])
        assert result == {
            "hello": {1: [0]},
            "world": {2: [0]},
        }

    def test_merge_overlapping_posting_lists(self):
        """Test merging posting lists with shared terms."""
        pl1 = {"hello": {1: [0, 2]}, "world": {1: [1]}}
        pl2 = {"hello": {2: [0]}, "there": {2: [1]}}
        result = merge_posting_lists([pl1, pl2])
        assert result["hello"] == {1: [0, 2], 2: [0]}
        assert result["world"] == {1: [1]}
        assert result["there"] == {2: [1]}

    def test_merge_same_doc_same_term(self):
        """Test merging when same term appears in same doc."""
        pl1 = {"hello": {1: [0]}}
        pl2 = {"hello": {1: [2]}}
        result = merge_posting_lists([pl1, pl2])
        assert result["hello"] == {1: [0, 2]}

    def test_merge_single_posting_list(self):
        """Test merging a single posting list."""
        pl = {"hello": {1: [0]}, "world": {1: [1]}}
        result = merge_posting_lists([pl])
        assert result == pl

    def test_merge_empty_list(self):
        """Test merging empty list of posting lists."""
        result = merge_posting_lists([])
        assert result == {}

    def test_merge_three_lists(self):
        """Test merging three posting lists."""
        pl1 = {"a": {1: [0]}}
        pl2 = {"a": {2: [0]}, "b": {2: [1]}}
        pl3 = {"a": {3: [0]}, "b": {3: [1]}, "c": {3: [2]}}
        result = merge_posting_lists([pl1, pl2, pl3])
        assert result["a"] == {1: [0], 2: [0], 3: [0]}
        assert result["b"] == {2: [1], 3: [1]}
        assert result["c"] == {3: [2]}


class TestGetTermGaps:
    """Tests for get_term_gaps function."""

    def test_single_occurrence(self):
        """Test term with only one occurrence."""
        posting_list = {"hello": {1: [0]}}
        result = get_term_gaps(posting_list, "hello")
        assert result == {}

    def test_two_occurrences(self):
        """Test term with two occurrences."""
        posting_list = {"hello": {1: [0, 3]}}
        result = get_term_gaps(posting_list, "hello")
        assert result == {1: [3]}

    def test_multiple_occurrences(self):
        """Test term with many occurrences."""
        posting_list = {"apple": {1: [0, 2, 4, 6, 8]}}
        result = get_term_gaps(posting_list, "apple")
        assert result == {1: [2, 2, 2, 2]}

    def test_irregular_gaps(self):
        """Test term with irregular gap pattern."""
        posting_list = {"word": {1: [0, 1, 5, 6, 15]}}
        result = get_term_gaps(posting_list, "word")
        assert result == {1: [1, 4, 1, 9]}

    def test_nonexistent_term(self):
        """Test retrieving gaps for term not in posting list."""
        posting_list = {"hello": {1: [0]}}
        result = get_term_gaps(posting_list, "goodbye")
        assert result == {}

    def test_multiple_documents(self):
        """Test term gaps across multiple documents."""
        posting_list = {
            "term": {
                1: [0, 3, 5],
                2: [1, 4],
            }
        }
        result = get_term_gaps(posting_list, "term")
        assert result == {1: [3, 2], 2: [3]}

    def test_document_with_single_occurrence(self):
        """Test when a document has a term with single occurrence."""
        posting_list = {
            "word": {
                1: [0, 2, 4],
                2: [5],
            }
        }
        result = get_term_gaps(posting_list, "word")
        # Document 2 should not be in results since it has only one occurrence
        assert result == {1: [2, 2]}


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_posting_list_and_gaps(self):
        """Test creating posting list and analyzing gaps."""
        text = "a b a c a d a e a"
        posting_list = text_to_posting_list(text, doc_id=1)
        gaps = get_term_gaps(posting_list, "a")
        
        assert posting_list["a"] == {1: [0, 2, 4, 6, 8]}
        assert gaps == {1: [2, 2, 2, 2]}

    def test_multiple_docs_with_gaps(self):
        """Test posting list with multiple documents and gap analysis."""
        docs = [
            "apple banana apple orange apple",
            "banana apple banana",
        ]
        posting_list = documents_to_posting_list(docs)
        apple_gaps = get_term_gaps(posting_list, "apple")
        
        assert posting_list["apple"] == {0: [0, 2, 4], 1: [1]}
        assert apple_gaps == {0: [2, 2]}  # Document 1 has only one occurrence

    def test_full_workflow(self):
        """Test complete workflow from documents to gap analysis."""
        documents = [
            "the quick brown fox jumps over the lazy dog",
            "the fox is quick",
        ]
        
        # Create posting list
        posting_list = documents_to_posting_list(documents)
        
        # Verify term appears in both docs
        assert "the" in posting_list
        assert 0 in posting_list["the"]
        assert 1 in posting_list["the"]
        
        # Analyze gaps for a term
        gaps = get_term_gaps(posting_list, "the")
        assert 0 in gaps  # Document 0 has multiple occurrences
        assert 1 not in gaps  # Document 1 has only one occurrence


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_very_long_document(self):
        """Test with a very long document."""
        text = " ".join(["word"] * 1000)
        result = text_to_posting_list(text, doc_id=1)
        assert len(result["word"][1]) == 1000

    def test_many_unique_terms(self):
        """Test with many unique terms."""
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        result = text_to_posting_list(text, doc_id=1)
        assert len(result) == 100

    def test_mixed_case_consistency(self):
        """Test that mixed case normalization is consistent."""
        texts = [
            "Hello World Hello",
            "hello world hello",
            "HELLO WORLD HELLO",
        ]
        results = [text_to_posting_list(t, doc_id=1) for t in texts]
        
        # All should produce the same result
        assert results[0] == results[1] == results[2]

    def test_positions_are_zero_indexed(self):
        """Verify that positions start at 0."""
        result = text_to_posting_list("first second third", doc_id=1)
        assert result["first"][1][0] == 0
        assert result["second"][1][0] == 1
        assert result["third"][1][0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
