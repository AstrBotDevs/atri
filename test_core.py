import sys

sys.path.append("/home/soulter/AstrBot")
from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI
from core.starter import ATRIMemoryStarter


async def main():
    gemini_provider = ProviderGoogleGenAI(
        provider_config={
            "id": "new_gemini(googlegenai原生)_6",
            "type": "googlegenai_chat_completion",
            "enable": True,
            "key": ["AIzaSyBcusI1DFIQZw3TbgDT1Fbj8JJQimPFMJ4"],
            "api_base": "https://dynamic-halva-76bb38.netlify.app/",
            "model_config": {"model": "gemini-2.0-flash"},
        },
        provider_settings={
            "unique_session": False,
            "rate_limit": {"time": 60, "count": 30, "strategy": "stall"},
            "reply_prefix": "",
            "forward_threshold": 10,
            "id_whitelist": [],
            "wl_ignore_admin_on_group": True,
            "wl_ignore_admin_on_friend": True,
            "id_whitelist_log": True,
            "enable_id_white_list": True,
            "reply_with_mention": True,
            "reply_with_quote": True,
            "path_mapping": [],
            "segmented_reply": {
                "enable": False,
                "only_llm_result": True,
                "interval": "1.5,2.0",
                "regex": ".*?[。？！~…]+|.+$",
                "seg_prompt": "",
                "interval_method": "log",
                "log_base": 2.6,
                "words_count_threshold": 70,
                "content_cleanup_rule": "",
            },
            "no_permission_reply": True,
            "empty_mention_waiting": True,
            "friend_message_needs_wake_prefix": False,
            "ignore_bot_self_message": False,
        },
        db_helper=None,
        persistant_history=True,
        default_persona=None,
    )

    memory_layer = ATRIMemoryStarter(
        data_dir_path="test_mem",
        llm_provider=gemini_provider,
    )
    await memory_layer.initialize()

    while True:
        user_input = input("You: ")
        if user_input == "/exit":
            break
        response = await gemini_provider.text_chat(
            prompt=user_input,
            system_prompt="",
        )
        print(f"Response: {response.completion_text}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
