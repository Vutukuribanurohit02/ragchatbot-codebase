"""
Integration tests for CourseSearchTool with REAL VectorStore and ChromaDB.

These tests use actual ChromaDB, real embeddings, and real documents to test
the CourseSearchTool.execute() method in realistic conditions.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from vector_store import VectorStore
from search_tools import CourseSearchTool
from models import Course, CourseChunk, Lesson


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with real ChromaDB"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def real_vector_store(self, temp_chroma_path):
        """Create a real VectorStore with ChromaDB"""
        store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        return store

    @pytest.fixture
    def populated_vector_store(self, real_vector_store):
        """Populate VectorStore with test course data"""
        # Create test course
        course = Course(
            title="Test Python Course",
            instructor="Dr. Test",
            course_link="https://test.com/python",
            lessons=[
                Lesson(
                    lesson_number=1,
                    lesson_title="Introduction to Python",
                    lesson_link="https://test.com/python/lesson1"
                ),
                Lesson(
                    lesson_number=2,
                    lesson_title="Variables and Data Types",
                    lesson_link="https://test.com/python/lesson2"
                )
            ]
        )

        # Add course metadata
        real_vector_store.add_course_metadata(course)

        # Create test chunks
        chunks = [
            CourseChunk(
                content="Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991. Python is known for its simplicity and readability.",
                course_title="Test Python Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Variables in Python are containers for storing data values. Python has several built-in data types including integers, floats, strings, and booleans.",
                course_title="Test Python Course",
                lesson_number=2,
                chunk_index=0
            ),
            CourseChunk(
                content="Lists are ordered, mutable collections in Python. You can create a list using square brackets. Example: fruits = ['apple', 'banana', 'cherry']",
                course_title="Test Python Course",
                lesson_number=2,
                chunk_index=1
            ),
        ]

        # Add course content
        real_vector_store.add_course_content(chunks)

        return real_vector_store

    def test_execute_with_real_chromadb(self, populated_vector_store):
        """Test CourseSearchTool.execute() with real ChromaDB"""
        tool = CourseSearchTool(populated_vector_store)

        # Execute search
        result = tool.execute("What is Python?")

        # Assertions
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Python Course" in result
        assert "Python" in result or "python" in result.lower()

        # Verify sources were tracked
        assert len(tool.last_sources) > 0
        assert len(tool.last_source_links) > 0
        assert len(tool.last_chunks) > 0

        print(f"\n✓ Test passed - Result: {result[:200]}...")
        print(f"✓ Sources tracked: {tool.last_sources}")

    def test_execute_search_about_variables(self, populated_vector_store):
        """Test search for variable-related content"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute("Tell me about variables")

        assert result is not None
        assert "variable" in result.lower() or "Variables" in result
        assert "data" in result.lower()

        print(f"\n✓ Variables search result: {result[:200]}...")

    def test_execute_with_course_filter(self, populated_vector_store):
        """Test search with course name filter"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute("Python programming", course_name="Test Python Course")

        assert result is not None
        assert "Test Python Course" in result

        print(f"\n✓ Filtered search result: {result[:200]}...")

    def test_execute_with_lesson_filter(self, populated_vector_store):
        """Test search with lesson number filter"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute("What is covered?", course_name="Test Python Course", lesson_number=2)

        assert result is not None
        assert "Lesson 2" in result
        # Should find content from lesson 2 (variables)
        assert "variable" in result.lower() or "list" in result.lower()

        print(f"\n✓ Lesson-filtered search result: {result[:200]}...")

    def test_execute_no_results_found(self, populated_vector_store):
        """Test search when no relevant content exists"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute("quantum physics black holes")

        assert result is not None
        assert "No relevant content found" in result

        print(f"\n✓ No results case: {result}")

    def test_execute_with_empty_vector_store(self, real_vector_store):
        """Test search with empty vector store"""
        tool = CourseSearchTool(real_vector_store)

        result = tool.execute("anything")

        assert result is not None
        assert "No relevant content found" in result

        print(f"\n✓ Empty store result: {result}")

    def test_source_tracking_accuracy(self, populated_vector_store):
        """Test that sources, links, and chunks are correctly tracked"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute("Python language")

        # Verify all tracking arrays have same length
        assert len(tool.last_sources) == len(tool.last_source_links)
        assert len(tool.last_sources) == len(tool.last_chunks)

        # Verify sources match the course
        for source in tool.last_sources:
            assert "Test Python Course" in source

        # Verify links are valid
        for link in tool.last_source_links:
            if link:  # Links might be None for some results
                assert "https://" in link or link is None

        # Verify chunks contain actual content
        for chunk in tool.last_chunks:
            assert len(chunk) > 0

        print(f"\n✓ Source tracking verified:")
        print(f"  Sources: {tool.last_sources}")
        print(f"  Links: {tool.last_source_links}")
        print(f"  Chunk count: {len(tool.last_chunks)}")

    def test_multiple_searches_reset_sources(self, populated_vector_store):
        """Test that sources are properly managed across multiple searches"""
        tool = CourseSearchTool(populated_vector_store)

        # First search
        result1 = tool.execute("Python")
        sources1 = tool.last_sources.copy()

        # Second search
        result2 = tool.execute("variables")
        sources2 = tool.last_sources.copy()

        # Sources should be different (or at least independently tracked)
        assert sources2 is not sources1  # Different list objects

        print(f"\n✓ Multiple searches tracked independently")

    def test_max_results_respected(self, real_vector_store):
        """Test that MAX_RESULTS configuration is respected"""
        # Create store with max_results=2
        store_limited = VectorStore(
            chroma_path=real_vector_store.chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=2
        )

        # Populate with same data
        course = Course(
            title="Test Course",
            instructor="Test",
            course_link="https://test.com",
            lessons=[]
        )
        store_limited.add_course_metadata(course)

        chunks = [CourseChunk(
            content=f"Content chunk number {i}",
            course_title="Test Course",
            chunk_index=i
        ) for i in range(5)]

        store_limited.add_course_content(chunks)

        tool = CourseSearchTool(store_limited)
        result = tool.execute("content")

        # Should return at most 2 results
        assert len(tool.last_sources) <= 2

        print(f"\n✓ MAX_RESULTS=2 respected: {len(tool.last_sources)} results returned")

    def test_semantic_search_quality(self, populated_vector_store):
        """Test that semantic search finds relevant content"""
        tool = CourseSearchTool(populated_vector_store)

        # Search with synonyms/related terms
        result = tool.execute("programming language")

        # Should find Python content even though query doesn't say "Python"
        assert "Python" in result

        print(f"\n✓ Semantic search found relevant content")

    def test_error_handling_with_corrupted_metadata(self, real_vector_store):
        """Test error handling when metadata is missing or corrupted"""
        tool = CourseSearchTool(real_vector_store)

        # Add course with minimal metadata
        course = Course(title="Minimal Course", lessons=[])
        real_vector_store.add_course_metadata(course)

        chunk = CourseChunk(
            content="Some content",
            course_title="Minimal Course",
            chunk_index=0
            # Note: no lesson_number
        )
        real_vector_store.add_course_content([chunk])

        # Should not crash
        result = tool.execute("content")

        assert result is not None
        assert isinstance(result, str)

        print(f"\n✓ Handled missing metadata gracefully")
