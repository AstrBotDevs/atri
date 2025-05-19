import pytest
import os
from core.starter import ATRIMemoryStarter
from core.provider.llm.openai_source import ProviderOpenAI


class TestCore:
    @classmethod
    def setup_class(cls):
        cls.starter = None
        cls.provider = None

    @pytest.mark.asyncio
    async def test_initialize(self):
        self.provider = ProviderOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("OPENAI_MODEL"),
            base_url=os.environ.get("OPENAI_API_BASE_URL"),
        )
        self.starter = ATRIMemoryStarter(
            data_dir_path="test_core",
            llm_provider=self.provider,
        )
        await self.starter.initialize()

    @pytest.mark.asyncio
    async def test_core(self):
        if not self.starter:
            await self.test_initialize()
        s = self.starter

        # --- Add Graph Memory ---
        atri_user_id = "atri_001"
        atri_user_name = "ATRI"
        sample_dialog_1 = """ATRI: 我今天去北海道的小樽玩了！
Assistant: 哇！好羡慕呀！北海道的雪景一定很美吧？
ATRI: 是啊！我去了天狗山，看到很多雪山和滑雪场。
Assistant: 天狗山是个滑雪胜地！你滑雪了吗？
ATRI: 没有，我不会滑雪，只是在山上拍了很多照片，小樽很美呢。
Assistant: 小樽是个浪漫的地方！你那里吃了什么？
ATRI: 吃了刺生...但是拉肚子了，我可能对海鲜过敏。
Assistant: 哎呀，真可惜！注意身体哦！下次去北海道可以试试其他美食，比如拉面和奶酪蛋糕。"""

        # Another domain dialog that is not related to the first one
        sample_dialog_2 = """ATRI: 高数好难呀... 我今天期中考试只考了 42 分。
Assistant: 哇，真低啊...你复习了吗？
ATRI: 肯定复习了，但是考试的时候脑子一片空白，完全想不起来。
Assistant: 你复习的内容和考试的内容不一样吗？
ATRI: 好像有点不一样... 考前重点看了课后题，结果考试好多证明题，我根本没准备。
Assistant: 嗯，证明题确实需要单独训练。你有试着总结过公式或者常见题型吗？
ATRI: 有做一些笔记，但都是抄的，没太理解，感觉只是完成任务一样。
Assistant: 我懂了，下次我们可以一起推公式、练证明，不是背下来，而是理解它怎么来的，这样考试才不容易慌。
"""

        # test memory update
        sample_dialog_3 = """ATRI: 我有一个外号，叫高性能机器人
Assistant: 哈哈，为什么叫这个外号呢？
ATRI: 因为我是高性能的嘛！
Assistant: 哦，原来如此！你觉得自己最厉害的地方是什么？
ATRI: 我觉得我最厉害的地方是吃螃蟹！我最喜欢吃螃蟹了！
"""
        # test memory update
        sample_dialog_4 = """ATRI: 我一点都不喜欢吃螃蟹，因为我对螃蟹过敏！
Assistant: 哦，原来如此！那你喜欢吃什么呢？
ATRI: 我喜欢吃关东煮！
Assistant: 关东煮很好吃！你最喜欢的配料是什么？
ATRI: 我最喜欢吃白萝卜和鸡蛋！
"""
        dialogs = [sample_dialog_1, sample_dialog_2, sample_dialog_3, sample_dialog_4]
        for dialog in dialogs:
            text = await s.summarizer.summarize(dialog)
            print(f"Summarized Text: {text}")
            await s.graph_memory.add_to_graph(
                text=text,
                user_id=atri_user_id,
                group_id=None,
                username=atri_user_name,
            )
        # --- test edges ---
        ret = s.kuzu_graph_store.get_passage_edges(
            filter={
                "user_id": atri_user_id,
            }
        )
        ret_ls = list(ret)
        print(f"Passage Edges: {ret_ls}")
        assert len(ret_ls) > 0

        # --- Search Graph Memory ---
        filters = {
            "user_id": atri_user_id,
        }
        results = await s.graph_memory.search_graph(
            "北海道真好玩！",
            num_to_retrieval=5,
            filters=filters,
        )
        print(f"Results: {results}")
        assert len(results) > 0

    @classmethod
    def teardown_class(cls):
        import shutil

        if os.path.exists("test_core"):
            shutil.rmtree("test_core")
