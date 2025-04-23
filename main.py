import json
from astrbot.api import logger
from astrbot.api.provider import ProviderRequest
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult  # noqa
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger  # noqa
from .core.starter import ATRIMemoryStarter
from collections import defaultdict

PLUGIN_DATA_DIR = StarTools.get_data_dir("atri")


@register("atri", "Soulter", "ATRI - My Dear Moments", "0.0.1")
class ATRIPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.user_counter = defaultdict(int)
        # 阈值
        self.sum_threshold = 5

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        self.llm_provider = self.context.provider_manager.curr_provider_inst
        self.memory_layer = ATRIMemoryStarter(
            data_dir_path=PLUGIN_DATA_DIR,
            llm_provider=self.llm_provider,
        )
        await self.memory_layer.initialize()

    @filter.on_llm_request()
    async def requesting(self, event: MessageEventResult, req: ProviderRequest):
        """处理请求事件"""
        results = await self.memory_layer.graph_memory.search_graph(req.prompt)
        if results:
            req.system_prompt += (
                "\n\nHere are related memories between you and user:\n" + str(results)
            )

    @filter.after_message_sent()
    async def after_message(self, event: AstrMessageEvent):
        """处理消息事件"""
        uid = event.unified_msg_origin
        self.user_counter[uid] += 1
        if self.user_counter[uid] >= self.sum_threshold:
            cid = await self.context.conversation_manager.get_curr_conversation_id(uid)
            conv = await self.context.conversation_manager.get_conversation(uid, cid)
            logger.info(
                f"User {uid} has sent {self.user_counter[uid]} messages. Summarizing conversation."
            )
            self.user_counter[uid] = 0
            text = await self.memory_layer.summarizer.summarize(json.loads(conv.history))
            await self.memory_layer.graph_memory.add_to_graph(text)
            logger.info("Added to graph.")

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
