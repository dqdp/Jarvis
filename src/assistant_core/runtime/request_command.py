from __future__ import annotations

from assistant_core.config.settings import Settings
from assistant_core.domain.conversations import RecentMessagesQuery
from assistant_core.runtime.agent_runtime import RuntimeTurnCommand


class RuntimeTurnCommandBuilder:
    def __init__(self, *, conversation_store, settings: Settings) -> None:
        self._conversation_store = conversation_store
        self._settings = settings

    async def build(self, request_record) -> RuntimeTurnCommand:
        messages = await self._conversation_store.load_recent_messages(
            RecentMessagesQuery(conversation_id=request_record.conversation_id, limit=1000),
        )
        user_message = next(
            message for message in messages if message.message_id == request_record.user_message_id
        )
        conversation = await self._conversation_store.get_conversation(request_record.conversation_id)
        if conversation is None:
            raise KeyError("conversation not found")
        return RuntimeTurnCommand(
            request_id=request_record.request_id,
            conversation_id=request_record.conversation_id,
            user_message_id=request_record.user_message_id,
            user_id=self._settings.app.default_user_id,
            user_input=user_message.content,
            active_project_namespace=conversation.active_project_namespace,
            current_message_sensitivity=user_message.sensitivity,
            model_profile=request_record.metadata.get("model_profile", "local_main"),
            loop_strategy=request_record.metadata.get("loop_strategy", "memory_augmented_answer"),
            working_directory=request_record.metadata.get("working_directory"),
            permission_mode=self._settings.permissions.mode,
            metadata=dict(request_record.metadata),
        )
