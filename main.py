from collections import defaultdict
import asyncio

from quart import request

from astrbot.api import logger  # noqa
from astrbot.api.event import (
    AstrMessageEvent,
    filter,
)  # noqa
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.provider.entities import LLMResponse
from astrbot.dashboard.server import Response

from .core.starter import ATRIMemoryStarter

PLUGIN_DATA_DIR = StarTools.get_data_dir("atri")


@register("atri", "Soulter", "ATRI - My Dear Moments", "0.0.1")
class ATRIPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.user_counter = defaultdict(int)
        # 阈值
        self.sum_threshold = 10
        self.dialogs = defaultdict(list)
        self._user_queues = defaultdict(asyncio.Queue)
        self._user_workers = {}
        self.running = True

        self.context.register_web_api(
            "/alkaid/ltm/graph", self.api_get_graph, ["GET"], "获取记忆图数据"
        )
        self.context.register_web_api(
            "/alkaid/ltm/user_ids", self.api_get_user_ids, ["GET"], "获取所有用户ID"
        )

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

    @staticmethod
    def parse_identifier(event: AstrMessageEvent):
        name = event.get_sender_name()
        user_id = event.get_sender_id()
        if name == user_id:
            return name
        elif not name:
            return user_id
        else:
            return name

    async def get_persona_id(self, umo: str):
        """获取当前对话的人格ID"""
        curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
        conversation = await self.context.conversation_manager.get_conversation(
            umo, curr_cid
        )
        persona_id = conversation.persona_id if conversation else None

        # 如果persona_id为空且不是明确设置为"[%None]"，则使用默认人格
        if not persona_id and persona_id != "[%None]":
            if self.context.provider_manager.selected_default_persona:
                persona_id = self.context.provider_manager.selected_default_persona[
                    "name"
                ]
            else:
                persona_id = "Me"

        return persona_id

    @staticmethod
    def _task_done_callback(task):
        """处理task完成的回调函数"""
        try:
            task.result()
        except Exception as e:
            logger.error(f"Task failed with error: {e}")

    async def _summary_worker(self, uid: str):
        """处理单个用户的总结任务的worker"""
        queue = self._user_queues[uid]
        while True:
            try:
                # 等待队列中的任务
                dialog, event = await queue.get()
                await self._handle_summary(dialog, uid, event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker for user {uid} encountered error: {e}")
            finally:
                queue.task_done()

    async def _start_worker(self, uid: str):
        """启动用户的worker任务"""
        if uid not in self._user_workers:
            worker = asyncio.create_task(self._summary_worker(uid))
            self._user_workers[uid] = worker
            worker.add_done_callback(lambda _: self._user_workers.pop(uid, None))

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
        if self.user_counter[uid] >= self.sum_threshold and self.running:
            logger.info(
                f"User {uid} has sent {self.user_counter[uid]} messages. Queuing summary task."
            )
            await self._start_worker(uid)
            await self._user_queues[uid].put((self.dialogs[uid].copy(), event))
            self.user_counter[uid] = 0
            self.dialogs[uid].clear()

    async def _handle_summary(self, dialog: list, uid: str, event: AstrMessageEvent):
        """处理对话总结的异步任务"""
        try:
            dialog_str = "\n".join(dialog)
            text = await self.memory_layer.summarizer.summarize(dialog_str)
            logger.debug(f"Summarized text: {text}")

            if "%None%" in text.strip():
                logger.info("没有符合总结的内容，跳过这轮总结。")
                # rollback
                self.dialogs[uid][0:0] = dialog
                self.user_counter[uid] = len(self.dialogs[uid])
                return
            elif "%Hold%" in text.strip():
                logger.info("对话话题不完整，继续观察。")
                # rollback
                self.dialogs[uid][0:0] = dialog
                self.user_counter[uid] = len(self.dialogs[uid])
                return

            await self.memory_layer.graph_memory.add_to_graph(
                text=text,
                user_id=str(event.get_sender_id()),
                group_id=str(event.get_group_id()),
                username=event.get_sender_name(),
            )
            logger.info("Added to graph.")
        except Exception as e:
            logger.error(f"Summary processing failed: {str(e)}")

    @filter.on_llm_response()
    async def after_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """将LLM回复加入到dialogs"""
        umo = event.unified_msg_origin
        persona_id = await self.get_persona_id(umo)
        self.dialogs[umo].append(
            f"{persona_id}: {response.result_chain.get_plain_text()}"
        )

    async def terminate(self):
        """清理"""
        self.running = False
        for worker in self._user_workers.values():
            if not worker.done():
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

        for queue in self._user_queues.values():
            try:
                while not queue.empty():
                    queue.get_nowait()
                    queue.task_done()
            except asyncio.QueueEmpty:
                pass

        self._user_workers.clear()
        self._user_queues.clear()
