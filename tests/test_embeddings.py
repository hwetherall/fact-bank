"""
Tests for the embeddings and deduplication module.

Run with: pytest tests/test_embeddings.py -v
"""

import pytest
import numpy as np

from src.utils.embeddings import (
    cosine_similarity,
    cosine_similarity_matrix,
    find_duplicate_pairs,
)


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors is 1.0."""
        vec = [1.0, 2.0, 3.0]
        
        similarity = cosine_similarity(vec, vec)
        
        assert similarity == pytest.approx(1.0, rel=1e-6)
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        
        assert similarity == pytest.approx(0.0, abs=1e-6)
    
    def test_opposite_vectors(self):
        """Test similarity of opposite vectors is -1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        
        assert similarity == pytest.approx(-1.0, rel=1e-6)
    
    def test_zero_vector(self):
        """Test similarity with zero vector is 0.0."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        
        assert similarity == 0.0


class TestCosineSimilarityMatrix:
    """Tests for batch similarity computation."""
    
    def test_identity_diagonal(self):
        """Test that diagonal of similarity matrix is all 1.0."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        
        matrix = cosine_similarity_matrix(embeddings)
        
        for i in range(3):
            assert matrix[i, i] == pytest.approx(1.0, rel=1e-6)
    
    def test_symmetry(self):
        """Test that similarity matrix is symmetric."""
        embeddings = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        
        matrix = cosine_similarity_matrix(embeddings)
        
        assert matrix[0, 1] == pytest.approx(matrix[1, 0], rel=1e-6)
        assert matrix[0, 2] == pytest.approx(matrix[2, 0], rel=1e-6)
        assert matrix[1, 2] == pytest.approx(matrix[2, 1], rel=1e-6)


class TestFindDuplicatePairs:
    """Tests for duplicate detection."""
    
    def test_no_duplicates(self):
        """Test with no duplicates (orthogonal vectors)."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        
        pairs = find_duplicate_pairs(embeddings, threshold=0.85)
        
        assert len(pairs) == 0
    
    def test_finds_duplicates(self):
        """Test finding duplicate pairs."""
        # Very similar vectors
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],  # Very similar to first
            [0.0, 1.0, 0.0],  # Different
        ]
        
        pairs = find_duplicate_pairs(embeddings, threshold=0.85)
        
        assert len(pairs) == 1
        assert pairs[0][0] == 0
        assert pairs[0][1] == 1
        assert pairs[0][2] > 0.85
    
    def test_single_embedding(self):
        """Test with single embedding returns no pairs."""
        embeddings = [[1.0, 2.0, 3.0]]
        
        pairs = find_duplicate_pairs(embeddings, threshold=0.85)
        
        assert len(pairs) == 0
    
    def test_empty_embeddings(self):
        """Test with empty list returns no pairs."""
        pairs = find_duplicate_pairs([], threshold=0.85)
        
        assert len(pairs) == 0
    
    def test_threshold_affects_results(self):
        """Test that threshold changes results."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.4, 0.0],  # Moderately similar
        ]
        
        # Normalize for proper cosine similarity
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        embeddings[0] = [x / norm1 for x in embeddings[0]]
        embeddings[1] = [x / norm2 for x in embeddings[1]]
        
        # With high threshold - no duplicates
        pairs_high = find_duplicate_pairs(embeddings, threshold=0.95)
        
        # With low threshold - finds duplicates  
        pairs_low = find_duplicate_pairs(embeddings, threshold=0.5)
        
        # Low threshold should find more or equal duplicates
        assert len(pairs_low) >= len(pairs_high)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

