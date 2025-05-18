from astrbot.api import logger
from astrbot.api.provider import ProviderRequest
from astrbot.api.event import (
    filter,
    AstrMessageEvent,
    MessageEventResult,
    ResultContentType,
)  # noqa
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger  # noqa
from astrbot.dashboard.server import Response
from .core.starter import ATRIMemoryStarter
from collections import defaultdict
from quart import request

PLUGIN_DATA_DIR = StarTools.get_data_dir("atri")


@register("atri", "Soulter", "ATRI - My Dear Moments", "0.0.1")
class ATRIPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.user_counter = defaultdict(int)
        # 阈值
        self.sum_threshold = 10
        self.dialogs = defaultdict(list)  # umo -> history
        self.context.register_web_api("/alkaid/ltm/graph", self.api_get_graph, ["GET"], "获取记忆图数据")
        self.context.register_web_api("/alkaid/ltm/user_ids", self.api_get_user_ids, ["GET"], "获取所有用户ID")

    async def api_get_graph(self):
        # params
        user_id = request.args.get("user_id", None)
        group_id = request.args.get("group_id", None)
        filter = {}
        if user_id:
            filter["user_id"] = user_id
        if group_id:
            filter["group_id"] = group_id
        result = await self.memory_layer.graph_memory.get_graph(filter)
        return Response().ok(data=result).__dict__

    async def api_get_user_ids(self):
        result = await self.memory_layer.graph_memory.get_user_ids()
        return Response().ok(data=result).__dict__

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        self.llm_provider = self.context.provider_manager.curr_provider_inst
        self.memory_layer = ATRIMemoryStarter(
            data_dir_path=PLUGIN_DATA_DIR,
            llm_provider=self.llm_provider,
        )
        await self.memory_layer.initialize()


    @filter.on_llm_request()
    async def requesting(self, event: AstrMessageEvent, req: ProviderRequest):
        """处理请求事件"""
        filters = {
            "user_id": str(event.get_sender_id()),
        }
        if event.get_group_id():
            filters["group_id"] = str(event.get_group_id())
        results = await self.memory_layer.graph_memory.search_graph(
            req.prompt,
            num_to_retrieval=5,
            filters=filters,
        )
        if results:
            req.system_prompt += (
                "\n\nHere are related memories between you and user:\n" + str(results)
            )

    def parse_identifier(self, event: AstrMessageEvent):
        name = event.get_sender_name()
        user_id = event.get_sender_id()
        if name == user_id:
            return name
        elif not name:
            return user_id
        else:
            return name

    # @filter.after_message_sent()
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def after_message(self, event: AstrMessageEvent):
        """处理消息事件"""
        if not event.message_str:  # TODO: 处理多模态信息
            return
        # result = event.get_result()
        # TODO: streaming result?
        # if not result or result.result_content_type != ResultContentType.LLM_RESULT:
        #     return
        uid = event.unified_msg_origin
        identifier = self.parse_identifier(event)
        message = event.message_str.replace("\n", " ")
        self.dialogs[uid].append(f"User({identifier}): {message}")
        # self.dialogs[uid].append(f"Me: {result.get_plain_text()}")

        self.user_counter[uid] += 1
        if self.user_counter[uid] >= self.sum_threshold:
            logger.info(
                f"User {uid} has sent {self.user_counter[uid]} messages. Summarizing conversation."
            )
            self.user_counter[uid] = 0
            dialog = self.dialogs[uid]
            dialog_str = "\n".join(dialog)
            text = await self.memory_layer.summarizer.summarize(dialog_str)
            logger.debug(f"Summarized text: {text}")
            if "%None%" in text.strip():
                logger.info("没有符合总结的内容，跳过这轮总结。")
                self.dialogs[uid].clear()
                return
            elif "%Hold%" in text.strip():
                logger.info("对话话题不完整，继续观察。")
                return
            await self.memory_layer.graph_memory.add_to_graph(
                text=text,
                user_id=str(event.get_sender_id()),
                group_id=str(event.get_group_id()),
                username=event.get_sender_name(),
            )
            logger.info("Added to graph.")
            self.dialogs[uid].clear()

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
