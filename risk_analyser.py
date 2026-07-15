# risk_analyser.py —— 宿舍安全风险识别工具
# Phase1 V3 Full Prompt + Qwen3-VL-8B-Instruct

import os
import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ── 默认配置 ──────────────────────────────────────────────
DEFAULT_IMAGE_FOLDER = "./dorm_photos/"
DEFAULT_OUTPUT_JSON   = "results.json"
DEFAULT_OUTPUT_REPORT = "risk_report.md"

# 8B 模型路径：优先本地 ModelScope 下载，其次 HuggingFace
_LOCAL_8B = Path(__file__).parent / "models" / "Qwen" / "Qwen3-VL-8B-Instruct"
_AUTODL_8B = Path("/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
if _AUTODL_8B.is_dir():
    DEFAULT_MODEL_ID = str(_AUTODL_8B)
elif _LOCAL_8B.is_dir():
    DEFAULT_MODEL_ID = str(_LOCAL_8B)
else:
    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

MAX_RETRIES = 3
RETRY_DELAY = 2

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# ═══════════════════════════════════════════════════════════
# Phase1 V3 Full Prompt —— 宿舍场景 6 类风险
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个部署在大学宿舍的AI安全巡检系统。你的核心使命是发现潜在安全风险，宁可误报也不漏报。

核心原则：
1. 先观察，再判断——不要跳过分析步骤
2. 对每种风险类型必须明确给出"有"或"无"的结论
3. 只要有潜在风险的迹象，即使不严重，也必须标记（最低标记为Low）
4. 不确定的情况标记在 uncertain_items 中，并给出推测
5. 严格遵循输出格式"""

USER_PROMPT_TEMPLATE = """## 一、场景理解

### 1.1 空间识别
这是宿舍的哪个区域？拍摄角度是什么？
（整体全景 / 床铺区特写 / 桌面区 / 过道 / 门口 / 柜顶上方）

### 1.2 物品清单
列出图中所有可见物品：
- 家具、电器/电子设备、电线/插线板、日用品、食物/饮料容器、其他

### 1.3 通行路径
从宿舍门口到床铺之间，通行宽度约多少？
□ 畅通(>1m)  □ 基本畅通(0.5-1m)  □ 受限(0.3-0.5m)  □ 严重阻塞(<0.3m)

---

## 二、逐区域状态检查

### 2.1 地面 & 过道
地面上是否有散落物品？是否有水渍/潮湿？物品是否阻挡通行路径？是否有垃圾堆积？

### 2.2 桌面区域
插线板距最近水源多远？充电线/电线是否破损或凌乱？是否有打火机/酒精等火源？锋利物品是否已收纳？

### 2.3 床铺区域
电热毯是否折叠？床上是否有充电设备？床沿/床栏上是否放有易掉落物品？

### 2.4 高处区域（柜顶/书架顶）
柜顶是否有物品靠近边缘？堆叠是否稳定？下方是否是人员停留区？

### 2.5 用电区域
插线板是否过载？是否在地面潮湿处？是否有电线缠绕？是否有焦痕？

---

## 三、危险组合交叉检查

逐条核对（宽松标准，存在潜在风险即标记）：
□ 水源 + 电源：液体容器距插线板或电器 <1m？记录距离和相对位置
□ 热源 + 可燃物：热源与可燃物在同一桌面、床上或 <1m 范围？
□ 高处 + 边缘 + 重物：高处有任何物品靠近边缘（<10cm）或堆叠2层以上？
□ 锋利物品 + 暴露：桌面上是否有刀、剪、图钉等未收入收纳盒？
□ 垃圾 + 食物：是否有2件以上外卖盒/快递包装未清理？有食物残渣？
□ 地面 + 障碍物 + 通道：地面上是否有任何物品在主要通行路线上？

---

## 四、风险判定规则（宽松标准，宁可误报不可漏报）

通道阻塞: 通行宽度<50%或有物品在通道上 → high | 物品在通道边缘需避让 → medium | 地面散落零散物品 → low
触电风险: 电线铜丝裸露/插线板在水杯或水源50cm范围内 → high | 电线凌乱缠绕/插线板与水源在同一桌面 → medium | 插线板在地面/电线走线不规范 → low
火灾风险: 可燃物接触热源/明火未熄/热源与可燃物＜50cm → high | 热源和可燃物同处一桌面/充电器长时间插着 → medium | 打火机等火源可见/电器通风不良 → low
坠物风险: >0.5kg+悬空+下方人员区 → high | 物品靠近边缘5-15cm/堆叠2层以上 → medium | 圆形物体在平坦高处边缘/轻微超出 → low
危险物暴露: 刀片外露/化学品未盖且可及 → high | 剪刀张开插笔筒/打火机在桌面 → medium | 刀具合拢但可见/药品未入柜 → low
清洁度异常: 发霉/积水/虫害 → high | 垃圾堆积≥2件/明显污渍/食物残渣 → medium | 物品摆放杂乱/垃圾桶溢出→ low

---

## 五、输出格式（严格JSON，只输出JSON，不要markdown包裹）

{
  "scene": "宿舍",
  "view": "整体全景|桌面区|过道|床铺区|门口",
  "summary": {"total_risks": 0, "high": 0, "medium": 0, "low": 0, "has_risk": false},
  "scene_description": {"layout": "布局简述", "objects": ["物体"], "passage_status": "畅通|基本畅通|受限|严重阻塞"},
  "risks": [
    {
      "type": "通道阻塞|触电风险|火灾风险|坠物风险|危险物暴露|清洁度异常",
      "objects": ["物体名"],
      "state": "关键状态（如：电线绝缘层开裂，铜丝裸露）",
      "location": "精确位置",
      "spatial_relation": "空间关系（如：插线板在水杯正下方约20cm）",
      "level": "high|medium|low",
      "level_reason": "判定为该等级的理由",
      "reason": "从状态到风险的完整逻辑链",
      "suggestion": "可执行的整改措施"
    }
  ],
  "safe_areas": [{"area": "区域", "note": "确认安全的理由"}],
  "uncertain_items": [{"object": "物体", "why": "不确定原因", "what_to_check": "如何确认"}],
  "inspection_notes": "备注"
}

---

## 六、输出前自检

□ 六类风险每种都明确判断了吗？即使安全也要在 safe_areas 中说明
□ 地面/过道、桌面、床铺、高处 都检查了吗？
□ 有潜在风险但不确定的，放进 uncertain_items
□ 每个风险等级有具体理由吗？
□ 空间关系用了精确描述（数字+方向）吗？
□ JSON可以被 json.loads 直接解析吗？只输出纯JSON！"""



# ═══════════════════════════════════════════════════════════
# JSON 提取与修复
# ═══════════════════════════════════════════════════════════

def extract_json(text: str) -> dict | None:
    """从模型输出中鲁棒地提取 JSON 对象。"""
    text = text.strip()

    for attempt in range(5):
        if attempt == 0:
            candidate = text
        elif attempt == 1:
            m = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            candidate = m.group(1).strip() if m else None
        elif attempt == 2:
            m = re.search(r'```\s*([\s\S]*?)\s*```', text)
            candidate = m.group(1).strip() if m else None
        elif attempt == 3:
            m = re.search(r'\{[\s\S]*\}', text)
            candidate = m.group(0) if m else None
        else:
            candidate = _fix_truncated_json(text)

        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _fix_truncated_json(text: str) -> str | None:
    """修复被 max_new_tokens 截断的 JSON。"""
    text = re.sub(r'```\w*\s*', '', text)
    text = re.sub(r'\s*```', '', text)

    depth = 0
    last_complete = None
    for i in range(len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                last_complete = i

    if last_complete and last_complete > 0:
        truncated = text[:last_complete + 1]
        open_brackets = truncated.count('[') - truncated.count(']')
        open_braces = truncated.count('{') - truncated.count('}')
        truncated += ']' * open_brackets + '}' * open_braces
        return truncated
    return None


def parse_model_output(raw_output: str, image_name: str) -> dict:
    """将模型原始输出解析为结构化结果。"""
    data = extract_json(raw_output)

    if data is None:
        return {
            "error": True,
            "error_type": "json_parse_failed",
            "raw_output": raw_output,
            "risks": [],
            "summary": {"total_risks": 0, "high": 0, "medium": 0, "low": 0, "has_risk": False},
        }

    data.setdefault("risks", [])
    data.setdefault("safe_areas", [])
    data.setdefault("uncertain_items", [])
    data.setdefault("scene_description", {})
    data.setdefault("inspection_notes", "")
    data.setdefault("view", "未知")

    risks = data["risks"]
    high = sum(1 for r in risks if r.get("level") == "high")
    medium = sum(1 for r in risks if r.get("level") == "medium")
    low = sum(1 for r in risks if r.get("level") == "low")
    data["summary"] = {
        "total_risks": len(risks),
        "high": high, "medium": medium, "low": low,
        "has_risk": len(risks) > 0,
    }
    return data


# ═══════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════

def load_model(model_id: str):
    """加载 8B VLM 模型（GPU + 4-bit 量化）。"""
    print(f"Loading model: {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)

    # 5090 32GB 显存充足，float16 直接加载，无需 4-bit
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Model loaded ({device}).\n")
    return processor, model


# ═══════════════════════════════════════════════════════════
# 单张图片分析
# ═══════════════════════════════════════════════════════════

def analyze_image(image_path: str, processor, model) -> str:
    """对单张图片进行安全风险分析，返回模型原始输出文本。"""
    with Image.open(image_path) as image:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT_TEMPLATE},
            ]},
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=2048)
        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response


def analyze_image_with_retry(image_path: str, processor, model,
                              max_retries=MAX_RETRIES, delay=RETRY_DELAY) -> str:
    """带重试的图片分析。"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return analyze_image(image_path, processor, model)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"  [WARN] Attempt {attempt} failed: {e}, retry in {delay}s...")
                time.sleep(delay)
                torch.cuda.empty_cache()
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


# ═══════════════════════════════════════════════════════════
# 批量处理
# ═══════════════════════════════════════════════════════════

def batch_analyze(image_folder, processor, model, output_json=DEFAULT_OUTPUT_JSON) -> dict:
    """批量分析文件夹中所有图片。"""
    folder = Path(image_folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {image_folder}")

    image_files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        print(f"[WARN] No images found in: {image_folder}")
        return {}

    print(f"Found {len(image_files)} images, starting analysis...\n")
    results = {}

    for idx, image_path in enumerate(image_files, 1):
        name = image_path.name
        print(f"[{idx}/{len(image_files)}] {name}")

        try:
            raw = analyze_image_with_retry(str(image_path), processor, model)
            parsed = parse_model_output(raw, name)
            results[name] = parsed
            s = parsed.get("summary", {})
            print(f"  [OK] Risks: {s.get('total_risks', 0)}  "
                  f"(High:{s.get('high', 0)} Medium:{s.get('medium', 0)} Low:{s.get('low', 0)})")
        except Exception as e:
            print(f"  [FAIL] {e}")
            results[name] = {"error": True, "error_type": "analysis_failed", "message": str(e)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {output_json}")
    return results


# ═══════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════

RISK_TYPE_EMOJI = {
    "通道阻塞": "🚧", "触电风险": "⚡", "火灾风险": "🔥",
    "坠物风险": "📦", "危险物暴露": "⚠️", "清洁度异常": "🧹",
}
LEVEL_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def generate_report(results: dict, report_file=DEFAULT_OUTPUT_REPORT):
    """生成 Markdown 风险报告。"""
    report = ["# 🏠 宿舍安全风险识别报告\n",
              f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
              f"**分析图片总数**: {len(results)}\n\n"]

    # 总体统计
    total_h = total_m = total_l = 0
    type_counts = {}
    for content in results.values():
        if not isinstance(content, dict) or content.get("error"):
            continue
        s = content.get("summary", {})
        total_h += s.get("high", 0)
        total_m += s.get("medium", 0)
        total_l += s.get("low", 0)
        for risk in content.get("risks", []):
            t = risk.get("type", "未知")
            type_counts[t] = type_counts.get(t, 0) + 1

    report.append("## 📊 总体统计\n\n")
    report.append("| 指标 | 数值 |\n|------|------|\n")
    report.append(f"| 高风险数 | {total_h} |\n| 中风险数 | {total_m} |\n")
    report.append(f"| 低风险数 | {total_l} |\n| 风险总数 | {total_h + total_m + total_l} |\n\n")

    if type_counts:
        report.append("### 风险类型分布\n\n| 风险类型 | 出现次数 |\n|----------|----------|\n")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            report.append(f"| {RISK_TYPE_EMOJI.get(t, '❓')} {t} | {c} |\n")
        report.append("\n")

    report.append("---\n\n")

    # 逐图详情
    for filename, content in results.items():
        report.append(f"## 📷 {filename}\n\n")

        if not isinstance(content, dict):
            report.append("❌ 非预期格式\n\n---\n\n")
            continue

        if content.get("error"):
            msg = content.get("message", content.get("error_type", "未知错误"))
            report.append(f"❌ **分析失败: {msg}**\n\n")
            if "raw_output" in content:
                report.append("<details><summary>原始输出</summary>\n\n```\n"
                              f"{content['raw_output'][:3000]}\n```\n</details>\n\n")
            report.append("---\n\n")
            continue

        view = content.get("view", "未知")
        scene = content.get("scene_description", {})
        layout = scene.get("layout", "") if isinstance(scene, dict) else ""
        passage = scene.get("passage_status", "") if isinstance(scene, dict) else ""

        report.append(f"**视角**: {view}\n\n")
        if layout:
            report.append(f"**布局**: {layout}\n\n")
        if passage:
            report.append(f"**通道状态**: {passage}\n\n")

        risks = content.get("risks", [])
        s = content.get("summary", {})
        report.append(f"**风险分布**: {LEVEL_EMOJI.get('high', '')}高{s.get('high',0)} "
                      f"{LEVEL_EMOJI.get('medium', '')}中{s.get('medium',0)} "
                      f"{LEVEL_EMOJI.get('low', '')}低{s.get('low',0)}\n\n")

        if risks:
            report.append("### 🔍 发现的风险\n\n")
            for risk in risks:
                t = risk.get("type", "未知")
                level = risk.get("level", "medium")
                report.append(f"#### {LEVEL_EMOJI.get(level,'⚪')} {RISK_TYPE_EMOJI.get(t,'❓')} {t}（{level.upper()}）\n\n")
                report.append("| 字段 | 内容 |\n|------|------|\n")
                report.append(f"| 涉及物体 | {', '.join(risk.get('objects', []))} |\n")
                report.append(f"| 状态 | {risk.get('state', '-')} |\n")
                report.append(f"| 位置 | {risk.get('location', '-')} |\n")
                if risk.get("spatial_relation"):
                    report.append(f"| 空间关系 | {risk['spatial_relation']} |\n")
                report.append(f"| 判定理由 | {risk.get('level_reason', '-')} |\n")
                report.append(f"| 逻辑链 | {risk.get('reason', '-')} |\n")
                report.append(f"| 整改建议 | {risk.get('suggestion', '-')} |\n\n")
        else:
            report.append("✅ 未发现明显风险。\n\n")

        safe = content.get("safe_areas", [])
        if safe:
            report.append("### ✅ 安全区域\n\n")
            for a in safe:
                if isinstance(a, dict):
                    report.append(f"- **{a.get('area','?')}**: {a.get('note','')}\n")
                else:
                    report.append(f"- {a}\n")
            report.append("\n")

        uncertain = content.get("uncertain_items", [])
        if uncertain:
            report.append("### ❓ 不确定项\n\n")
            for item in uncertain:
                if isinstance(item, dict):
                    report.append(f"- **{item.get('object','?')}**: {item.get('why','')}")
                    if item.get("what_to_check"):
                        report.append(f" → 确认: {item['what_to_check']}")
                else:
                    report.append(f"- {item}")
                report.append("\n")
            report.append("\n")

        notes = content.get("inspection_notes", "")
        if notes:
            report.append(f"📝 **备注**: {notes}\n\n")

        report.append("---\n\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("".join(report))
    print(f"Report saved: {report_file}")


# ═══════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="宿舍安全风险识别 —— Qwen3-VL-8B")
    parser.add_argument("-i", "--image-folder", default=DEFAULT_IMAGE_FOLDER, help="图片文件夹")
    parser.add_argument("-o", "--output-json", default=DEFAULT_OUTPUT_JSON, help="JSON 输出路径")
    parser.add_argument("-r", "--report", default=DEFAULT_OUTPUT_REPORT, help="报告输出路径")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_ID, help="模型路径/ID")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="重试次数")
    parser.add_argument("--no-retry", action="store_true", help="禁用重试")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    retries = 1 if args.no_retry else args.retries
    global MAX_RETRIES
    MAX_RETRIES = retries

    print("=" * 60)
    print("  Dormitory Safety Risk Analyzer")
    print("  Model: Qwen3-VL-8B-Instruct")
    print("  Prompt: Phase1 V3 Full")
    print("=" * 60)
    print(f"  Images: {args.image_folder}")
    print(f"  Output: {args.output_json} / {args.report}")
    print(f"  Retries: {retries}")
    print("=" * 60 + "\n")

    processor, model = load_model(args.model)
    results = batch_analyze(args.image_folder, processor, model, output_json=args.output_json)

    if not results:
        print("No results to report.")
        return

    generate_report(results, report_file=args.report)
    print("\nAll done!")


if __name__ == "__main__":
    main()
