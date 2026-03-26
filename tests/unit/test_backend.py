"""Tests for OpenAICompatibleLLM backend using mocked AsyncOpenAI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from syntheta.exceptions import AuthenticationError, ModelNotFoundError


class TestOpenAICompatibleLLM:
    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True), pytest.raises(
            AuthenticationError, match="No API key"
        ):
            from syntheta.llm.backend import OpenAICompatibleLLM

            OpenAICompatibleLLM(api_key_env="NONEXISTENT_KEY")

    def test_explicit_api_key(self):
        from syntheta.llm.backend import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(api_key="sk-test-key", model="gpt-4o-mini")
        assert llm.model == "gpt-4o-mini"

    def test_resolve_model_roles(self):
        from syntheta.llm.backend import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(
            api_key="sk-test",
            model="cheap",
            response_model="strong",
        )
        assert llm._resolve_model("model") == "cheap"
        assert llm._resolve_model("response_model") == "strong"

    def test_response_model_fallback(self):
        from syntheta.llm.backend import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(api_key="sk-test", model="cheap")
        assert llm._resolve_model("response_model") == "cheap"

    @pytest.mark.asyncio
    async def test_complete_success(self):
        from syntheta.llm.backend import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(api_key="sk-test", model="test-model")

        # Mock the response
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_message = MagicMock()
        mock_message.content = "Hello, world!"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "test-model"

        llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await llm.complete(
            messages=[{"role": "user", "content": "Hi"}],
            stage="test",
        )

        assert result["content"] == "Hello, world!"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert llm.cost_tracker.total_tokens == 30

    @pytest.mark.asyncio
    async def test_complete_auth_error(self):
        import openai

        from syntheta.llm.backend import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(api_key="sk-bad", model="test")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Invalid key"}}

        llm._client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid key",
                response=mock_response,
                body={"error": {"message": "Invalid key"}},
            )
        )

        with pytest.raises(AuthenticationError):
            await llm.complete(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_complete_not_found_error(self):
        import openai

        from syntheta.llm.backend import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(api_key="sk-test", model="nonexistent")

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Not found"}}

        llm._client.chat.completions.create = AsyncMock(
            side_effect=openai.NotFoundError(
                message="Not found",
                response=mock_response,
                body={"error": {"message": "Not found"}},
            )
        )

        with pytest.raises(ModelNotFoundError):
            await llm.complete(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_complete_batch(self):
        from syntheta.llm.backend import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(api_key="sk-test", model="test")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10

        def make_response(content: str):
            msg = MagicMock()
            msg.content = content
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = mock_usage
            resp.model = "test"
            return resp

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return make_response(f"Response {call_count}")

        llm._client.chat.completions.create = mock_create

        results = await llm.complete_batch(
            [
                [{"role": "user", "content": "Q1"}],
                [{"role": "user", "content": "Q2"}],
            ],
            stage="test",
        )

        assert len(results) == 2
        assert results[0]["content"] == "Response 1"
        assert results[1]["content"] == "Response 2"
