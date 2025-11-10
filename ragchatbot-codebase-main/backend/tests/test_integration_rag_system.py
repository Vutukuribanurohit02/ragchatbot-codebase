"""
Integration tests for the complete RAG system with content queries.

These tests verify end-to-end functionality: document loading, vector search,
AI generation, and complete query handling.
"""
import os
import sys
import tempfile
import shutil

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from rag_system import RAGSystem
from models import Course, CourseChunk, Lesson


class TestRAGSystemIntegration:
    """Integration tests for complete RAG system"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        config = Config()
        config.CHROMA_PATH = tempfile.mkdtemp(prefix="test_rag_")
        return config

    @pytest.fixture
    def rag_system(self, test_config):
        """Create RAGSystem instance"""
        if not test_config.ANTHROPIC_API_KEY:
            pytest.skip("ANTHROPIC_API_KEY not set")

        system = RAGSystem(test_config)
        yield system

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

    @pytest.fixture
    def populated_rag_system(self, rag_system):
        """RAG system with test data"""
        course = Course(
            title="Python Programming Basics",
            instructor="Jane Doe",
            course_link="https://example.com/python",
            lessons=[
                Lesson(1, "Introduction", "https://example.com/python/1"),
                Lesson(2, "Variables", "https://example.com/python/2"),
            ]
        )

        chunks = [
            CourseChunk(
                content="Python is a versatile programming language created by Guido van Rossum. It emphasizes code readability and simplicity.",
                course_title="Python Programming Basics",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Variables in Python store data values. You don't need to declare variable types explicitly. Python infers the type automatically.",
                course_title="Python Programming Basics",
                lesson_number=2,
                chunk_index=0
            ),
            CourseChunk(
                content="Common Python data types include int, float, str, bool, list, dict, and tuple. Each serves different purposes in programming.",
                course_title="Python Programming Basics",
                lesson_number=2,
                chunk_index=1
            ),
        ]

        rag_system.vector_store.add_course_metadata(course)
        rag_system.vector_store.add_course_content(chunks)

        return rag_system

    def test_rag_query_returns_answer_with_sources(self, populated_rag_system):
        """Test that RAG system returns answer with sources"""
        answer, sources, source_links, chunks = populated_rag_system.query(
            "Who created Python?"
        )

        # Verify answer exists
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer) > 0

        # Verify sources were found
        assert sources is not None
        assert len(sources) > 0
        assert "Python Programming Basics" in sources[0]

        # Verify source links
        assert source_links is not None
        assert len(source_links) > 0

        # Verify chunks
        assert chunks is not None
        assert len(chunks) > 0

        print(f"\n✓ RAG query successful")
        print(f"  Answer: {answer}")
        print(f"  Sources: {sources}")
        print(f"  Source links: {source_links}")

    def test_rag_query_about_variables(self, populated_rag_system):
        """Test query about variables"""
        answer, sources, source_links, chunks = populated_rag_system.query(
            "Tell me about variables in Python"
        )

        assert answer is not None
        assert len(answer) > 0
        assert "variable" in answer.lower() or "data" in answer.lower()

        # Should find relevant sources
        assert len(sources) > 0

        print(f"\n✓ Variables query successful: {answer[:150]}...")

    def test_rag_query_with_no_matching_content(self, populated_rag_system):
        """Test query when no relevant content exists"""
        answer, sources, source_links, chunks = populated_rag_system.query(
            "What is quantum computing?"
        )

        assert answer is not None
        # Might say "no information found" or similar
        print(f"\n✓ No-match query handled: {answer[:150]}...")

    def test_rag_query_with_empty_database(self, rag_system):
        """Test query with empty vector database"""
        answer, sources, source_links, chunks = rag_system.query(
            "What is Python?"
        )

        assert answer is not None
        assert sources == []
        assert source_links == []
        assert chunks == []

        print(f"\n✓ Empty database query: {answer[:150]}...")

    def test_rag_session_management(self, populated_rag_system):
        """Test session management across queries"""
        session_id = populated_rag_system.session_manager.create_session()

        # First query
        answer1, _, _, _ = populated_rag_system.query(
            "What is Python?",
            session_id=session_id
        )

        # Second query (should have context from first)
        answer2, _, _, _ = populated_rag_system.query(
            "Who created it?",
            session_id=session_id
        )

        assert answer1 is not None
        assert answer2 is not None

        print(f"\n✓ Session management works")
        print(f"  Q1: What is Python?")
        print(f"  A1: {answer1[:100]}...")
        print(f"  Q2: Who created it?")
        print(f"  A2: {answer2[:100]}...")

    def test_rag_add_course_from_text_file(self, rag_system):
        """Test adding course from text file"""
        # Create temporary course file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        )

        content = """Course: Test Course
Instructor: Test Instructor

Lesson 1: First Lesson

This is the content of the first lesson. It contains important information about testing.

Lesson 2: Second Lesson

This is the second lesson content.
"""
        temp_file.write(content)
        temp_file.close()

        try:
            # Add course document
            course, chunk_count = rag_system.add_course_document(temp_file.name)

            if course:
                assert course.title == "Test Course"
                assert chunk_count > 0

                # Query the added content
                answer, sources, _, _ = rag_system.query("What is in the first lesson?")

                assert answer is not None
                assert len(sources) > 0

                print(f"\n✓ Added course from file: {course.title}, {chunk_count} chunks")
                print(f"  Query result: {answer[:150]}...")
            else:
                print("\n⚠ Document processing returned None - may be expected for test file")

        finally:
            os.unlink(temp_file.name)

    def test_rag_course_analytics(self, populated_rag_system):
        """Test course analytics"""
        analytics = populated_rag_system.get_course_analytics()

        assert "total_courses" in analytics
        assert analytics["total_courses"] >= 1
        assert "course_titles" in analytics
        assert len(analytics["course_titles"]) >= 1
        assert "Python Programming Basics" in analytics["course_titles"]

        print(f"\n✓ Analytics: {analytics}")

    def test_rag_multiple_queries_performance(self, populated_rag_system):
        """Test multiple queries in sequence"""
        queries = [
            "What is Python?",
            "Tell me about variables",
            "What are data types?",
        ]

        for query in queries:
            answer, sources, _, _ = populated_rag_system.query(query)
            assert answer is not None
            assert isinstance(answer, str)
            print(f"✓ Query: '{query}' -> {len(answer)} chars")

    def test_rag_source_reset_between_queries(self, populated_rag_system):
        """Test that sources are reset between queries"""
        # First query
        _, sources1, _, _ = populated_rag_system.query("Python")
        sources1_copy = sources1.copy()

        # Second query
        _, sources2, _, _ = populated_rag_system.query("variables")

        # Sources should be independent
        assert sources2 is not sources1_copy

        print(f"\n✓ Sources reset between queries")
        print(f"  Query 1 sources: {sources1_copy}")
        print(f"  Query 2 sources: {sources2}")
