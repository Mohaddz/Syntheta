"""Tests for ColumnMapper auto-detection and field mapping."""

from syntheta.schema.column_mapper import ColumnMapper


class TestFormatDetection:
    def test_alpaca_format(self):
        mapper = ColumnMapper()
        records = [{"instruction": "Do X", "input": "", "output": "Done X"}]
        samples = mapper.map(records)
        assert len(samples) == 1
        assert samples[0].instruction == "Do X"
        assert samples[0].response == "Done X"

    def test_alpaca_with_input(self):
        mapper = ColumnMapper()
        records = [{"instruction": "Summarize", "input": "Long text here", "output": "Summary"}]
        samples = mapper.map(records)
        assert "Long text here" in samples[0].instruction

    def test_sharegpt_format(self):
        mapper = ColumnMapper()
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"},
                ]
            }
        ]
        samples = mapper.map(records)
        assert samples[0].turns is not None
        assert len(samples[0].turns) == 2
        assert samples[0].turns[0].role == "user"
        assert samples[0].turns[1].role == "assistant"

    def test_openai_messages_format(self):
        mapper = ColumnMapper()
        records = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        ]
        samples = mapper.map(records)
        assert len(samples[0].turns) == 3
        assert samples[0].turns[0].role == "system"

    def test_dpo_format(self):
        mapper = ColumnMapper()
        records = [{"prompt": "Explain X", "chosen": "Good answer", "rejected": "Bad answer"}]
        samples = mapper.map(records)
        assert samples[0].instruction == "Explain X"
        assert samples[0].chosen == "Good answer"
        assert samples[0].rejected == "Bad answer"

    def test_qa_format(self):
        mapper = ColumnMapper()
        records = [{"question": "What is 2+2?", "answer": "4"}]
        samples = mapper.map(records)
        assert samples[0].instruction == "What is 2+2?"
        assert samples[0].response == "4"

    def test_rlvr_format(self):
        mapper = ColumnMapper()
        records = [{"question": "Solve 2+2", "info": {"answer": 4}}]
        samples = mapper.map(records)
        assert samples[0].instruction == "Solve 2+2"
        assert samples[0].info == {"answer": 4}

    def test_pretrain_format(self):
        mapper = ColumnMapper()
        records = [{"text": "Some long pretraining text content here."}]
        samples = mapper.map(records)
        assert samples[0].text == "Some long pretraining text content here."


class TestFuzzySynonymMapping:
    def test_prompt_maps_to_instruction(self):
        mapper = ColumnMapper()
        records = [{"prompt": "Do something", "custom_col": "some value"}]
        samples = mapper.map(records)
        assert samples[0].instruction == "Do something"
        # custom_col is not a synonym, goes to extra
        assert samples[0].extra.get("custom_col") == "some value"

    def test_completion_maps_to_response(self):
        mapper = ColumnMapper()
        records = [{"prompt": "Do something", "completion": "Done it"}]
        samples = mapper.map(records)
        assert samples[0].instruction == "Do something"
        assert samples[0].response == "Done it"

    def test_query_maps_to_instruction(self):
        mapper = ColumnMapper()
        records = [{"query": "Search this"}]
        samples = mapper.map(records)
        assert samples[0].instruction == "Search this"

    def test_reply_maps_to_response(self):
        mapper = ColumnMapper()
        records = [{"query": "Ask", "reply": "Answer here"}]
        samples = mapper.map(records)
        assert samples[0].response == "Answer here"


class TestUserOverride:
    def test_explicit_column_map(self):
        mapper = ColumnMapper(column_map={"my_q": "instruction", "my_a": "response"})
        records = [{"my_q": "Custom question", "my_a": "Custom answer"}]
        samples = mapper.map(records)
        assert samples[0].instruction == "Custom question"
        assert samples[0].response == "Custom answer"

    def test_override_takes_precedence(self):
        mapper = ColumnMapper(column_map={"prompt": "response"})
        records = [{"prompt": "This goes to response", "text": "fallback"}]
        samples = mapper.map(records)
        assert samples[0].response == "This goes to response"


class TestExtraFields:
    def test_unknown_columns_go_to_extra(self):
        mapper = ColumnMapper()
        records = [{"instruction": "test", "custom_score": 0.95, "metadata": {"k": "v"}}]
        samples = mapper.map(records)
        assert samples[0].extra["custom_score"] == 0.95
        assert samples[0].extra["metadata"] == {"k": "v"}

    def test_empty_records(self):
        mapper = ColumnMapper()
        assert mapper.map([]) == []
