"""测试 longmemeval_config.json 中所有模型是否可以正常调用。"""

import asyncio
import json
import os
import sys
from pathlib import Path

# 加载 .env
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from reme.utils import load_env  # noqa: E402

load_env()

CONFIG_PATH = REPO_ROOT / "longmemeval_config.json"
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


async def test_model(model_name: str, purpose: str) -> dict:
    """对单个模型发送简单请求，返回测试结果。"""
    import openai

    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"\n[{purpose}] 测试模型: {model_name}")
    print(f"  base_url: {BASE_URL}")

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "请用一句话回答：1+1等于几？"},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        answer = (response.choices[0].message.content or "").strip()
        print(f"  ✓ 调用成功，回复: {answer!r}")
        print(f"    用量: prompt={response.usage.prompt_tokens}, "
              f"completion={response.usage.completion_tokens}, "
              f"total={response.usage.total_tokens}")
        return {"model": model_name, "purpose": purpose, "status": "ok", "answer": answer}
    except Exception as e:
        print(f"  ✗ 调用失败: {type(e).__name__}: {e}")
        return {"model": model_name, "purpose": purpose, "status": "fail", "error": str(e)}


async def main():
    print("=== LongMemEval 模型调用测试 ===")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    models = config.get("model", {})
    if not models:
        print("longmemeval_config.json 中没有找到模型配置！")
        sys.exit(1)

    print(f"配置文件: {CONFIG_PATH}")
    print(f"共 {len(models)} 个角色，去重后测试:")

    # 去重（同一模型只测一次，但记录所有角色）
    seen: dict[str, list[str]] = {}
    for purpose, model_name in models.items():
        seen.setdefault(model_name, []).append(purpose)

    results = []
    for model_name, purposes in seen.items():
        purpose_label = " / ".join(purposes)
        result = await test_model(model_name, purpose_label)
        results.append(result)

    # 汇总
    print("\n" + "=" * 60)
    print("汇总结果:")
    print("=" * 60)
    all_ok = True
    for r in results:
        status = "✓ PASS" if r["status"] == "ok" else "✗ FAIL"
        print(f"  [{status}] {r['model']} ({r['purpose']})")
        if r["status"] != "ok":
            all_ok = False
            print(f"          错误: {r.get('error', 'unknown')}")

    if all_ok:
        print("\n所有模型调用测试通过！")
    else:
        print("\n部分模型调用失败，请检查上方错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
