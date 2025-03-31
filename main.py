import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace

# 各言語専用のエージェント
japanese_agent = Agent(
    name="japanese_agent",
    instructions="You must respond only in Japanese. Use this agent when the input is in Japanese.",
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You must respond only in Spanish. Use this agent when the input is in Spanish.",
)

english_agent = Agent(
    name="english_agent",
    instructions="You must respond only in English. Use this agent when the input is in English.",
)

# トリアージエージェント：入力された文章の言語を判定し、適切なエージェントにハンドオフする
triage_agent = Agent(
    name="triage_agent",
    instructions=(
        "受信したメッセージの言語を判定し、以下のルールに従ってハンドオフしてください:\n"
        "・日本語の場合 -> japanese_agent\n"
        "・スペイン語の場合 -> spanish_agent\n"
        "・英語の場合 -> english_agent\n"
        "・判定が困難な場合は、英語エージェントにフォールバックすること。"
    ),
    handoffs=[japanese_agent, spanish_agent, english_agent],
)

async def main():
    # 会話をトレースするためのID作成
    conversation_id = str(uuid.uuid4().hex[:16])

    # 初回の入力（会話履歴に追加）
    msg = input("こんにちは！私は日本語、英語、スペイン語を話すことができるエージェントです。今日は何をしましょう？ \n")
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # 毎回トリアージエージェントから実行し、最新のユーザー入力の言語判定を行う
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                triage_agent,  # ここを常に triage_agent にする
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        # 会話履歴にシステムの回答を追加
        inputs = result.to_input_list()
        print("\n")

        # 次のユーザー入力を取得
        user_msg = input("エンターキーを押してください: ")
        inputs.append({"content": user_msg, "role": "user"})
        # 以降は、次のループで再び triage_agent が言語判定を実施するので、
        # agent の更新は不要です。

if __name__ == "__main__":
    asyncio.run(main())
