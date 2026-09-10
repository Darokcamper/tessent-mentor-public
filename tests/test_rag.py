import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.rag_builder import load_rag, retrieve, VECTORSTORE_DIR

class TestVLSIRAG(unittest.TestCase):
    
    def test_rag_loading(self):
        """Test if RAG index files exist and load_rag() behaves correctly."""
        index_exists = (VECTORSTORE_DIR / "faiss_index.index").exists()
        loaded = load_rag()
        self.assertEqual(loaded, index_exists)
        
    def test_retrieval_fallback(self):
        """Test retrieve returns results if index exists, or empty list if not."""
        results = retrieve("lockup latch clock skew")
        self.assertIsInstance(results, list)
        
        index_exists = (VECTORSTORE_DIR / "faiss_index.index").exists()
        if index_exists:
            self.assertTrue(len(results) > 0)
            self.assertIn("source", results[0])
            self.assertIn("page", results[0])
            self.assertIn("text", results[0])
        else:
            self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
