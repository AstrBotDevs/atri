import json
import jsonlines
import os
import datetime
import asyncio
from tqdm.asyncio import tqdm as atqdm
import tqdm
from loguru import logger
from core.starter import ATRIMemoryStarter
from core.provider.llm.openai_source import ProviderOpenAI

with open("LongMemEval/data/longmemeval_s", "r") as f:
    datasets = json.load(f)

provider = ProviderOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    base_url=os.environ.get("OPENAI_API_BASE_URL"),
)


async def process_question(question, dir, jsonls):
    question_id = question["question_id"]
    sessions = question["haystack_sessions"]
    question_str = question["question"]
    haystack_dates = question["haystack_dates"]

    starter = ATRIMemoryStarter(
        data_dir_path=f"{dir}/{question_id}",
        llm_provider=provider,
    )
    await starter.initialize()

    # 会话处理保持同步，确保顺序
    answer_chat_summary = ""
    for i, session in tqdm.tqdm(
        enumerate(sessions),
        total=len(sessions),
        desc=f"Processing {question_id}",
        leave=False,
    ):
        text = "DATETIME: " + haystack_dates[i] + "\n"
        has_answer = False
        if len(session) == 0:
            continue
        for turn in session:
            text += f"{turn['role']}: {turn['content']}\n"
            has_answer = True if "has_answer" in turn else False
        # logger.info(f"Processing session {i}: {text}")
        for retry in range(5):
            try:
                summary = await starter.summarizer.summarize(text, add_time=False)
                if has_answer:
                    answer_chat_summary = summary
                break
            except Exception as e:
                logger.error(f"Error summarizing session {i}: {e}")
                continue
        for retry in range(5):
            try:
                await starter.graph_memory.add_to_graph(
                    summary,
                    user_id="atri",
                )
                break
            except Exception as e:
                logger.error(f"Error summarizing session {i}: {e}")
                continue
        logger.info(f"Added to graph: {summary}")

    with open(f"{dir}/{question_id}/done", "w") as f:
        f.write("done")

    for retry in range(5):
        try:
            result = await starter.graph_memory.search_graph(
                question_str,
                num_to_retrieval=5,
                filters={
                    "user_id": "atri",
                },
            )
            result_2 = await starter.graph_memory.vec_db_summary.retrieve(
                question_str,
                k=5,
                metadata_filters={
                    "user_id": "atri",
                },
            )
            result_3 = await starter.graph_memory.vec_db.retrieve(
                question_str,
                k=8,
                metadata_filters={
                    "user_id": "atri",
                },
            )
            break
        except Exception as e:
            logger.error(f"Error searching graph: {e}")
            continue

    # graph res
    related_res = ""
    cnt = 0
    for key, val in result.items():
        cnt += 1
        related_res += f"Related {cnt}: {val['text']}\n"
    system_prompt = (
        "You are a helpful assistant. "
        f"Related history of you and the user: {related_res} "
    )

    # summary res
    cnt = 0
    related_res2 = ""
    for val in result_2:
        cnt += 1
        related_res2 += f"Related {cnt}: {val.data['text']}\n"
    system_prompt2 = (
        "You are a helpful assistant. "
        f"Related history of you and the user: {related_res2} "
    )

    # fact res
    cnt = 0
    related_res3 = ""
    for val in result_3:
        cnt += 1
        related_res3 += f"Related {cnt}: {val.data['text']}\n"
    system_prompt3 = (
        "You are a helpful assistant. "
        f"Related history of you and the user: {related_res3} "
    )

    prompt = question_str

    for retry in range(5):
        try:
            response = await provider.text_chat(prompt, system_prompt=system_prompt)
            response2 = await provider.text_chat(prompt, system_prompt=system_prompt2)
            response3 = await provider.text_chat(prompt, system_prompt=system_prompt3)
            break
        except Exception as e:
            logger.error(f"Error in text chat: {e}")
            continue

    result = {
        "question_id": question_id,
        "question": question_str,
        "answer": question.get("answer"),
        "hypothesis": response.completion_text,
        "hypothesis_2": response2.completion_text,
        "hypothesis_3": response3.completion_text,
        "_answer_chat_summary": answer_chat_summary,
        "_search_result": related_res,
    }
    jsonls.write(result)
    return result


async def main():
    START = 0
    END = 1

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")

    dir = f"bm_lme/benchmark_longmemeval_{curr_time}_{os.environ.get('OPENAI_MODEL')}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    jsonls = jsonlines.open(f"{dir}/longmemeval_s.jsonl", mode="w")
    jsonls._flush = True

    # 异步并行处理所有问题，添加进度条
    tasks = [
        process_question(question, dir, jsonls) for question in datasets[START:END]
    ]
    await atqdm.gather(*tasks, desc="Processing all questions")

    jsonls.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
