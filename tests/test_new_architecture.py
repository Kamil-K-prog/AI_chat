import os
from datetime import datetime, timezone
from types import SimpleNamespace

# config.settings инициализируется во время импорта в нескольких рабочих модулях.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini")
os.environ.setdefault("OPENAI_API_KEY", "test-openai")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek")
os.environ.setdefault("GLM_API_KEY", "test-glm")
os.environ.setdefault("KIMI_API_KEY", "test-kimi")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter")
os.environ.setdefault("OPENROUTER_API_SECRET", "test-openrouter-secret")
os.environ.setdefault("SYSTEM_PROMPT", "test-system")
os.environ.setdefault("MEDIA_FOLDER", "/tmp/ai-chat-test-media")

import utils.types as t
from models.catalog import get_spec, list_specs
from models.converters import OpenAiHistoryConverter
from models.providers import make_model
from models.providers.openai import OpenAiProvider
from models.thinking import ThinkingPolicy
from models.tools import ToolRunner


def _msg(role: str, content: list[t.ContentItem]) -> t.Message:
    return t.Message(
        id=f"msg-{role}",
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def test_catalog_returns_registered_model_specs():
    spec = get_spec("deepseek-v4-flash")

    assert spec.provider == "openai"
    assert spec.api_key_setting == "DEEPSEEK_API_KEY"
    assert spec.base_url == "https://api.deepseek.com"
    assert any(item.name == "deepseek-v4-flash" for item in list_specs())


def test_make_model_builds_openai_provider_from_catalog():
    model = make_model("deepseek-v4-flash", system_prompt="system")

    assert isinstance(model, OpenAiProvider)
    assert model.spec.name == "deepseek-v4-flash"
    assert model.system_prompt == "system"


def test_openai_converter_maps_umf_messages_to_chat_completion_format():
    spec = get_spec("deepseek-v4-pro")
    converter = OpenAiHistoryConverter(spec)
    history = t.ChatData(messages=[
        _msg("system", [t.TextContent(text="system")]),
        _msg("user", [t.TextContent(text="hello")]),
        _msg("assistant", [
            t.ThoughtContent(text="private reasoning"),
            t.TextContent(text="visible answer"),
            t.ToolCallContent(tool_call=t.ToolCall(id="call-1", name="sum", args={"a": 1, "b": 2})),
        ]),
        _msg("tool", [
            t.ToolResultContent(tool_result=t.ToolResult(id="call-1", name="sum", text_content="3"))
        ]),
    ])

    native = converter.to_native(history)

    assert native[0] == {"role": "system", "content": "system"}
    assert native[1] == {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    assert native[2]["role"] == "assistant"
    assert native[2]["reasoning_content"] == "private reasoning"
    assert native[2]["content"] == [{"type": "text", "text": "visible answer"}]
    assert native[2]["tool_calls"][0]["function"]["name"] == "sum"
    assert native[3]["role"] == "tool"
    assert native[3]["tool_call_id"] == "call-1"


def test_openai_converter_maps_response_to_umf_assistant_message():
    spec = get_spec("deepseek-v4-pro")
    converter = OpenAiHistoryConverter(spec)
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                reasoning_content="reasoning",
                content="answer",
                tool_calls=[SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(name="sum", arguments='{"a": 1, "b": 2}'),
                )],
            )
        )]
    )

    delta = converter.from_native(response)

    assert len(delta) == 1
    assert delta[0].role == "assistant"
    assert [item.type for item in delta[0].content] == ["thought", "text", "tool_call"]
    assert delta[0].content[2].tool_call.name == "sum"
    assert delta[0].content[2].tool_call.args == {"a": 1, "b": 2}


def test_tool_runner_executes_tool_calls_and_returns_one_tool_message():
    def add(a: int, b: int) -> int:
        return a + b

    assistant = _msg("assistant", [
        t.ToolCallContent(tool_call=t.ToolCall(id="call-1", name="add", args={"a": 2, "b": 3}))
    ])

    result = ToolRunner().run(assistant, {"add": add})

    assert result is not None
    assert result.role == "tool"
    assert result.content[0].tool_result.id == "call-1"
    assert result.content[0].tool_result.text_content == "5"
    assert result.content[0].tool_result.is_error is False


def test_thinking_policy_preserved_prunes_completed_cycle_but_keeps_active_cycle():
    completed_assistant = _msg("assistant", [
        t.ThoughtContent(text="old thought"),
        t.TextContent(text="old answer"),
    ])
    active_assistant = _msg("assistant", [
        t.ThoughtContent(text="active thought", signature="sig"),
        t.ToolCallContent(tool_call=t.ToolCall(id="call-1", name="lookup", args={})),
    ])
    history = t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(thinking_mode="preserved")),
        messages=[
            _msg("user", [t.TextContent(text="first")]),
            completed_assistant,
            _msg("user", [t.TextContent(text="second")]),
            active_assistant,
        ],
    )

    pruned = ThinkingPolicy().apply(history)

    assert [item.type for item in pruned.messages[1].content] == ["text"]
    assert [item.type for item in pruned.messages[3].content] == ["thought", "tool_call"]
