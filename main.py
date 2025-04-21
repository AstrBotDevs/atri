from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult # noqa
from astrbot.api.star import Context, Star, register
from astrbot.api import logger # noqa


@register("atri", "Soulter", "ATRI - My Dear Moments", "0.0.1")
class ATRIPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
