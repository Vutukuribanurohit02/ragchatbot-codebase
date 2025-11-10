"""
Integration tests for AIGenerator calling CourseSearchTool.

These tests verify that AIGenerator correctly uses tool calling to execute
CourseSearchTool and handles the results properly.
"""
import os
import sys
import tempfile
import shutil

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from ai_generator import AIGenerator
from vector_store import VectorStore
from search_tools import CourseSearchTool, ToolManager
from models import Course, CourseChunk


class TestAIGeneratorToolCallingIntegration:
    """Integration tests for AIGenerator with real tool execution"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary ChromaDB directory"""
        temp_dir = tempfile.mkdtemp(prefix="test_chroma_ai_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def populated_vector_store(self, temp_chroma_path):
        """Create and populate VectorStore"""
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)

        course = Course(
            title="AI Fundamentals",
            instructor="Dr. Smith",
            course_link="https://test.com/ai",
            lessons=[]
        )
        store.add_course_metadata(course)

        chunks = [
            CourseChunk(
                content="Artificial Intelligence is the simulation of human intelligence by machines. AI systems can learn, reason, and solve problems.",
                course_title="AI Fundamentals",
                chunk_index=0
            ),
            CourseChunk(
                content="Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
                course_title="AI Fundamentals",
                chunk_index=1
            ),
        ]
        store.add_course_content(chunks)

        return store

    @pytest.fixture
    def real_tool_manager(self, populated_vector_store):
        """Create ToolManager with real CourseSearchTool"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        tool_manager.register_tool(search_tool)
        return tool_manager

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with real API key"""
        config = Config()
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("ANTHROPIC_API_KEY not set")
        return AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

    def test_ai_generator_calls_search_tool(self, ai_generator, real_tool_manager):
        """Test that AIGenerator correctly calls CourseSearchTool"""
        query = "What is artificial intelligence?"

        response = ai_generator.generate_response(
            query=query,
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        # Verify response was generated
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify tool was called
        sources = real_tool_manager.get_last_sources()
        assert sources is not None
        assert len(sources) > 0

        print(f"\n✓ AI called tool and generated response")
        print(f"  Response: {response[:200]}...")
        print(f"  Sources: {sources}")

    def test_ai_generator_uses_retrieved_context(self, ai_generator, real_tool_manager):
        """Test that AI uses context from tool execution"""
        query = "Explain machine learning"

        response = ai_generator.generate_response(
            query=query,
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        # Response should mention machine learning
        assert "machine learning" in response.lower() or "ml" in response.lower()

        # Should have retrieved context
        chunks = real_tool_manager.get_last_chunks()
        assert chunks is not None
        assert len(chunks) > 0

        print(f"\n✓ AI used retrieved context")
        print(f"  Response: {response}")

    def test_ai_generator_without_tools(self, ai_generator):
        """Test AIGenerator without tool calling"""
        query = "What is 2+2?"

        response = ai_generator.generate_response(query=query)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        print(f"\n✓ AI works without tools: {response}")

    def test_tool_execution_error_handling(self, ai_generator):
        """Test error handling when tool execution fails"""
        # Create broken tool manager
        broken_manager = ToolManager()

        # Register a tool but don't populate vector store (will cause errors)
        empty_store = VectorStore(tempfile.mkdtemp(), "all-MiniLM-L6-v2", max_results=5)
        broken_tool = CourseSearchTool(empty_store)
        broken_manager.register_tool(broken_tool)

        query = "Search for something"

        # Should not crash even if tool fails
        response = ai_generator.generate_response(
            query=query,
            tools=broken_manager.get_tool_definitions(),
            tool_manager=broken_manager
        )

        assert response is not None
        assert isinstance(response, str)

        print(f"\n✓ Handled tool errors gracefully: {response[:100]}...")
