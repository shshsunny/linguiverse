# 此为子会话模块，包括案例解释和交互解释的pipeline函数
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime, timedelta
import time
from openai import OpenAI
from copy import deepcopy
from pprint import pprint
from utils import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SD utils
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor
import random
import json
import time

import os
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

# FLUX utils
import requests
from http import HTTPStatus


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_CACHE_DIR = None # initialized in init_flux_pipeline

# image generation
DEFAULT_IMAGE_COUNT = 3  # 提高采样数量以便重排序
MAX_IMAGE_COUNT = 3      # 最大允许生成的图片数量

## Agent-type selection: language-agnostic (no FR-specific lexicons)

_st_model = None
def _get_st_model():
    global _st_model
    if _st_model is None:
        try:
            _st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=str(device))
        except Exception:
            # 兜底：返回 None 时跳过相似度打分
            _st_model = None
    return _st_model

# --- Lazy CLIP resources for image-text scoring ---
_clip_bundle = None
def _get_clip_bundle():
    """Lazily load CLIP model+processor on the detected device.
    Returns (processor, model). If loading fails, returns (None, None).
    """
    global _clip_bundle
    if _clip_bundle is not None:
        return _clip_bundle
    try:
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        model = CLIPModel.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        model = model.to(device)
        _clip_bundle = (processor, model)
    except Exception as e:
        print(f"[WARN] CLIP load failed: {e}. Using uniform fallback. To enable CLIP, ensure network or pre-download")
        _clip_bundle = (None, None)
    return _clip_bundle

# --- CLIP scoring helpers and quality gates ---
def _clip_probs_for_image_labels(image_path: str, labels: list[str]) -> dict:
    """Return softmax probs over labels for a single image using CLIP.
    If CLIP unavailable, return uniform distribution.
    """
    labels = [str(l).strip() for l in (labels or []) if str(l).strip()]
    if not labels:
        return {}
    processor, model = _get_clip_bundle()
    try:
        if processor is None or model is None:
            # uniform fallback
            p = 1.0 / len(labels)
            return {l: p for l in labels}
        img = Image.open(image_path).convert("RGB")
        inputs = processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits_per_image.squeeze(0)  # (num_labels)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()
        return {l: float(p) for l, p in zip(labels, probs)}
    except Exception as e:
        print(f"[WARN] CLIP infer failed: {e}. Using uniform fallback.")
        # robust fallback
        p = 1.0 / len(labels)
        return {l: p for l in labels}


def clip_score_image_against_goals(image_path: str, goals: list[str]) -> tuple[float, dict]:
    """Score image by CLIP against short text goals; return (max_prob, full_probs)."""
    goals = [g for g in (goals or []) if str(g).strip()]
    if not goals:
        return 0.0, {}
    probs = _clip_probs_for_image_labels(image_path, goals)
    best = max(probs.values()) if probs else 0.0
    return best, probs


def quality_gates_clip(image_path: str, primary_agent: str = 'human') -> dict:
    """Lightweight quality gates using CLIP zero-shot classification.
    Returns dict with flags and raw probabilities.
    """
    # Text presence gate
    text_labels = ["no visible text", "has visible text"]
    text_probs = _clip_probs_for_image_labels(image_path, text_labels)
    p_no = float(text_probs.get("no visible text", 0.5))
    p_has = float(text_probs.get("has visible text", 0.5))
    text_free = (p_no >= 0.5) and (p_no >= p_has + 0.1)

    # Agent presence gate
    agent_labels = ["human", "animal", "object"]
    ap = _clip_probs_for_image_labels(image_path, agent_labels)
    primary = (primary_agent or 'human').lower()
    p_primary = float(ap.get(primary, 0.0))
    top_label = max(ap, key=ap.get) if ap else None
    agent_present = (top_label == primary) and (p_primary >= 0.4)

    # Detect uniform-fallback distributions (neutralize gates when detected)
    def _is_uniform(d: dict) -> bool:
        if not d:
            return False
        vals = list(d.values())
        if not vals:
            return False
        m = sum(vals) / len(vals)
        return max(abs(v - m) for v in vals) <= 1e-6

    clip_fallback = False
    if _is_uniform(text_probs) and _is_uniform(ap):
        clip_fallback = True
        text_free = None
        agent_present = None

    return {
        "text_probs": text_probs,
        "agent_probs": ap,
        "text_free": (None if text_free is None else bool(text_free)),
        "agent_present": (None if agent_present is None else bool(agent_present)),
        "clip_fallback": clip_fallback,
    }


def build_text_goals_from_plan_or_context(subject: dict, meta: dict, example: str, plan: dict | None) -> list[str]:
    """Build a concise list (<=4) of text goals for CLIP scoring.
    Priority: plan.sense/required_cues/composition -> fallbacks from subject/meta/example.
    """
    goals: list[str] = []
    def add(x):
        s = str(x).strip()
        if s and s.lower() not in {g.lower() for g in goals}:
            goals.append(s)

    if plan:
        sense = (plan.get("sense") or "").strip()
        for cu in _ensure_list(plan.get("required_cues")):
            add(cu)
        if sense:
            add(sense)
        comp = plan.get("composition") or []
        if isinstance(comp, list) and comp:
            # take a short descriptor
            add(", ".join([str(c) for c in comp[:2]])[:80])

    # fallbacks
    if len(goals) < 2:
        add(subject.get('content', ''))
    if len(goals) < 3 and (meta is not None):
        try:
            add(meta.get('type', ''))
        except Exception:
            try:
                add(str(meta['type']))
            except Exception:
                pass
    if len(goals) < 4 and example:
        # keep only a brief phrase from example
        ex = str(example).strip()
        add(ex[:80])

    # limit to 4
    return goals[:4]

def determine_agent_types(subject, meta, example: str):
    """基于 metadata 与轻量启发式/嵌入，确定主体类型优先级。
    Returns: list[str] e.g., ['human'] or ['animal'] or ['object'] or ['human','animal']
    """
    word = (subject or {}).get('content', '') or ''
    wlow = word.lower()
    typ = (meta.get('type', '') or '').lower() if isinstance(meta, dict) else str(meta.get('type', '')).lower() if hasattr(meta, 'get') else ''

    # 1) 语言无关启发：优先依据词性（若提供）
    primary = None

    # 2) 词性/主题启发（支持多语言常见标签）
    if not primary:
        adj_tags = {'adjectif', 'adjective'}
        adv_tags = {'adverbe', 'adverb'}
        noun_tags = {'nom', 'noun'}
        verb_tags = {'verbe', 'verbe pronominal', 'verb'}
        if typ in adj_tags or typ in adv_tags:
            # 可修饰人或动物：保留双通道
            primary = 'human'
            candidate = ['human', 'animal']
        elif typ in noun_tags:
            # 名词：若无强指示，保守选 object（后续由嵌入纠偏）
            primary = 'object'
            candidate = ['object', 'human']
        elif typ in verb_tags:
            primary = 'human'
            candidate = ['human', 'object']
        else:
            candidate = []
    else:
        candidate = [primary, 'animal' if primary=='human' else 'human']

    # 3) 嵌入相似度兜底（若可用，语言无关）
    model = _get_st_model()
    if model is not None:
        anchors = {
            'human': "a human person in a social scene, human emotions, interaction",
            'animal': "an animal in its habitat, animal behavior",
            'object': "an inanimate object or thing, functional use, still life"
        }
        try:
            q = word if word else example
            q_emb = model.encode([q], normalize_embeddings=True)
            a_sent = list(anchors.values())
            a_emb = model.encode(a_sent, normalize_embeddings=True)
            import numpy as np
            sims = (q_emb @ a_emb.T)[0]
            order = np.argsort(-sims)
            ordered = [list(anchors.keys())[i] for i in order]
            # 将嵌入排序与候选合并去重
            merged = []
            for x in (candidate or []) + ordered:
                if x not in merged:
                    merged.append(x)
            return merged[:2] if merged else ordered[:2]
        except Exception:
            pass

    # 兜底：如果已有候选
    if candidate:
        return candidate
    # 最兜底：保持 human 优先
    return ['human', 'animal']

def plan_visual_semantics(subject, meta, example, responser):
    """使用 LLM 生成视觉语义计划（JSON）。包含主体类型、必选/可选线索、禁用项与构图建议。
    返回 dict；解析失败则抛出异常，由调用方处理回退。
    """
    concept = subject.get('content', '')
    typ = meta.get('type', '') if isinstance(meta, dict) else (meta.get('type', '') if hasattr(meta, 'get') else '')
    topic = meta.get('topic', '') if isinstance(meta, dict) else (meta.get('topic', '') if hasattr(meta, 'get') else '')
    guide = meta.get('guideword', '') if isinstance(meta, dict) else (meta.get('guideword', '') if hasattr(meta, 'get') else '')

    sys = (
        "Plan visual semantics for a text-to-image prompt. Output STRICT JSON only, no code fences, no extra text. "
        "Fields: sense, agent_type: {primary, secondary?}, required_cues (list of short phrases), optional_cues (list), "
        "forbidden_cues (list), composition (list of camera/lighting/DoF/style). "
        "Do not include any text/caption/sign/poster related cues."
    )
    user = (
        f"Concept: {concept}\nType: {typ}\nTopic: {topic}\nGuideword: {guide}\n"
        f"Example sentence: {example}\n"
        "Clarify the intended sense in 'sense' (one line). "
        "Choose agent_type.primary among ['human','animal','object'] and optional secondary if ambiguous. "
        "In required_cues, include: agent presence tokens, emotion/state, action/posture, social/environment context, clear visual cues to disambiguate meaning. "
        "In composition, include 2-3 items like 'medium shot', 'soft natural light', 'shallow depth of field'."
    )
    raw = get_response(context=[{"role":"system","content":sys},{"role":"user","content":user}], **responser).strip()
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip('`')
            # 若模型输出 ```json ... ```，做一次粗清理
            idx = cleaned.find('{')
            cleaned = cleaned[idx:] if idx != -1 else cleaned
        plan = json.loads(cleaned)
        # 规范化字段
        plan = {
            "sense": plan.get("sense", ""),
            "agent_type": plan.get("agent_type", {}),
            "required_cues": plan.get("required_cues", []) or [],
            "optional_cues": plan.get("optional_cues", []) or [],
            "forbidden_cues": plan.get("forbidden_cues", []) or [],
            "composition": plan.get("composition", []) or [],
        }
        return plan
    except Exception as e:
        raise RuntimeError(f"planner_json_parse_failed: {e}\nraw=\n{raw[:400]}")

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x if str(i).strip()]
    return [str(x)] if str(x).strip() else []

def compose_from_plan(plan: dict, k: int, agent_priority: list[str]) -> list[str]:
    """根据计划生成 k 条逗号关键词提示。强制包含 required_cues，各槽位尽量覆盖。
    保证无文本元素；对 forbidden 进行过滤。
    """
    primary = (plan.get("agent_type", {}) or {}).get("primary") or (agent_priority[0] if agent_priority else 'human')
    secondary = (plan.get("agent_type", {}) or {}).get("secondary") or (agent_priority[1] if agent_priority and len(agent_priority)>1 else None)
    req = _ensure_list(plan.get("required_cues"))
    opt = _ensure_list(plan.get("optional_cues"))
    forb = set([s.lower() for s in _ensure_list(plan.get("forbidden_cues"))])
    comp = _ensure_list(plan.get("composition"))

    # Agent canonical tokens
    agent_tokens = {
        'human': ["human", "person", "people", "portrait", "face", "crowd"],
        'animal': ["animal", "wildlife", "cow", "cattle", "bird", "horse", "farm"],
        'object': ["object", "item", "tool", "still life", "product"],
    }
    prim_tokens = agent_tokens.get(primary, agent_tokens['human'])
    sec_tokens = agent_tokens.get(secondary, []) if secondary else []

    def sanitize_tokens(tokens: list[str]) -> list[str]:
        bad = {"text","word","label","caption","subtitle","writing","handwriting","sign","poster","billboard","logo"}
        out = []
        for t in tokens:
            s = str(t).strip()
            if not s:
                continue
            if any(b in s.lower() for b in bad):
                continue
            out.append(s)
        return out

    req = sanitize_tokens(req)
    opt = sanitize_tokens(opt)
    comp = sanitize_tokens(comp)
    prim_tokens = sanitize_tokens(prim_tokens)
    sec_tokens = sanitize_tokens(sec_tokens)

    rng = random.Random()
    prompts = []
    for _ in range(k):
        parts = []
        # 强制主体可见
        if prim_tokens:
            parts.append(rng.choice(prim_tokens))
        # required_cues 尽量全部加入（短语）
        parts.extend(req[:])
        # 可选线索采样 1-2 个
        if opt:
            nopt = 1 if len(opt) < 2 else rng.choice([1,2])
            parts.extend(rng.sample(opt, k=nopt))
        # 构图采样 2 个
        if comp:
            parts.extend(rng.sample(comp, k=min(2, len(comp))))
        # 若有次主体，少量加入一个提示但不喧宾夺主
        if sec_tokens and rng.random() < 0.4:
            parts.append(rng.choice(sec_tokens))

        # 去重 / 过滤 forbidden
        seen = set()
        final = []
        forb_low = [f.lower() for f in forb]
        for p in parts:
            pl = p.lower()
            if pl in seen:
                continue
            if any(f in pl for f in forb_low):
                continue
            seen.add(pl)
            final.append(p)
        prompt = ", ".join(final)
        prompts.append(prompt)
    return prompts

def score_prompt_for_agent(prompt: str, agent: str) -> float:
    """启发式为提示词打分，衡量其与主体类型一致性。
    仅使用关键字与简单惩罚，范围大致在 [-1, 3]。
    """
    p = (prompt or '').lower()
    score = 0.0
    # 通用惩罚：文字相关词
    for ban in ["text", "word", "label", "caption", "subtitle", "writing", "handwriting"]:
        if ban in p:
            score -= 1.0
    # 过度拥挤/广告/标牌等
    for neg in ["sign", "poster", "billboard", "logo"]:
        if neg in p:
            score -= 0.6

    if agent == 'human':
        pos = ["human", "person", "people", "man", "woman", "girl", "boy", "face", "portrait", "crowd", "audience"]
        for kw in pos:
            if kw in p:
                score += 0.6
        # 若出现特定动物词，略降分
        for kw in ["cow", "cattle", "dog", "cat", "bird", "horse"]:
            if kw in p:
                score -= 0.3
    elif agent == 'animal':
        pos = ["animal", "wildlife", "cow", "cattle", "ox", "bull", "dog", "cat", "bird", "horse", "zoo", "farm"]
        for kw in pos:
            if kw in p:
                score += 0.6
        # 人类强词降分
        for kw in ["portrait", "face", "selfie", "crowd", "audience", "people", "person"]:
            if kw in p:
                score -= 0.3
    else:  # object
        # 鼓励物体与场景细节
        pos = ["object", "tool", "item", "still life", "close-up", "macro", "product"]
        for kw in pos:
            if kw in p:
                score += 0.6
        # 人类/动物词惩罚
        for kw in ["person", "people", "face", "portrait", "animal", "cow", "dog", "cat", "bird"]:
            if kw in p:
                score -= 0.4
    return score

# --------- Stable Diffusion utility ------------- #
"""
# 初始化SD模型管道
def init_sd_pipeline(model_cache_dir, model_name="stabilityai/stable-diffusion-xl-base-1.0", max_attempts=3, device="cpu"):
    # 确保D盘模型目录存在
    cache_dir=model_cache_dir
    global MODEL_CACHE_DIR
    MODEL_CACHE_DIR = model_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    for attempt in range(max_attempts):
        try:
            # 直接从HuggingFace下载到指定目录
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
            print("successfully loaded SD pipeline")
            # 根据设备自动设置数据类型
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipeline = pipeline.to(device=device, dtype=dtype)

            # 禁用安全检查器
            pipeline.safety_checker = None

            return pipeline

        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"模型下载尝试 {attempt+1} 失败，重试中...")
                continue
            raise RuntimeError(f"模型初始化失败: {str(e)}")

def generate_images(prompts, pipeline, output_dir: str = "generated_images"):
    使用SD生成指定数量的图片
    Args:
        prompt: 图片生成提示词
        pipeline: SD pipeline instance from state
    Returns:
        list: 生成的图片路径列表

    try:
        # 参数校验
        prompts = prompts[:min(len(prompts), MAX_IMAGE_COUNT)]
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        for i, prompt in enumerate(prompts):
            image = pipeline(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=512,   
                height=512
            ).images[0]
            
            img_path = os.path.join(output_dir, f"{int(time.time())}_{i}.png")
            image.save(img_path)
            images.append(img_path)
            
        return images
    except Exception as e:
        print(f"Error generating images: {e}")
        return []
"""

# ----------- FLUX utility ---------------------- #
def init_flux_pipeline(api_key: str, model_name):
    """初始化FLUX模型的配置信息
    Args:
        api_key: 百炼API密钥
        model_name: FLUX模型名称，支持 flux-schnell, flux-dev
    Returns:
        dict: FLUX配置字典
    """
    return {
        "api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
        "api_key": api_key,
        "model": model_name,
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    }

def generate_images_flux(prompts, flux_cfg, target_lang, output_dir: str = "generated_images"):
    """调用百炼FLUX API生成图像（异步方式）
    Args:
        prompts: 图片生成提示词列表
        flux_cfg: FLUX配置字典
    Returns:
        list: 生成的图片路径列表
    """
    
    os.makedirs(output_dir, exist_ok=True)
    images = []

    # 确保headers中所有值都是字符串类型，并过滤掉非标准HTTP header
    clean_headers = {}
    standard_headers = ['Authorization', 'Content-Type', 'Accept', 'User-Agent', 'X-DashScope-Async']
    
    for key, value in flux_cfg.get('headers', {}).items():
        if key in standard_headers:
            clean_headers[key] = str(value) if not isinstance(value, str) else value
    
    # 确保必要的headers存在
    if 'Authorization' not in clean_headers:
        clean_headers['Authorization'] = f"Bearer {flux_cfg['api_key']}"
    if 'Content-Type' not in clean_headers:
        clean_headers['Content-Type'] = "application/json"
    # 添加异步调用标识
    clean_headers['X-DashScope-Async'] = "enable"

    for i, prompt in enumerate(prompts[:MAX_IMAGE_COUNT]):
        try:
            # 构建请求数据
            payload = {
                "model": flux_cfg['model'],
                "input": {
                    "prompt": prompt
                },
                "parameters": {
                    "size": "1024*1024",
                    "steps": 4 if flux_cfg['model'] == "flux-schnell" else 30
                }
            }

            # 提交异步任务
            response = requests.post(
                flux_cfg['api_url'], 
                headers=clean_headers, 
                json=payload
            )
            
            if response.status_code == HTTPStatus.OK:
                result = response.json()
                task_id = result['output']['task_id']
                print(f"[INFO] FLUX任务提交成功 - Prompt {i+1}, Task ID: {task_id}")
                
                # 轮询任务状态
                task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                task_headers = {
                    'Authorization': clean_headers['Authorization']
                }
                
                max_retries = 30  # 最大重试次数
                retry_count = 0
                
                while retry_count < max_retries:
                    time.sleep(2)  # 等待2秒再查询
                    task_response = requests.get(task_url, headers=task_headers)
                    
                    if task_response.status_code == HTTPStatus.OK:
                        task_result = task_response.json()
                        task_status = task_result['output']['task_status']
                        
                        if task_status == 'SUCCEEDED':
                            # 任务成功，下载图片
                            image_url = task_result['output']['results'][0]['url']
                            img_response = requests.get(image_url)
                            
                            if img_response.status_code == 200:
                                img_path = os.path.join(output_dir, f"{target_lang}-{int(time.time())}-{i}.png")
                                with open(img_path, 'wb') as f:
                                    f.write(img_response.content)
                                images.append(img_path)
                                print(f"[INFO] 图片已保存: {img_path}")
                            break
                            
                        elif task_status == 'FAILED':
                            print(f"[ERROR] 任务失败 - Prompt {i+1}: {task_result.get('output', {}).get('message', 'Unknown error')}")
                            break
                            
                        elif task_status in ['PENDING', 'RUNNING']:
                            retry_count += 1
                            print(f"[INFO] 任务进行中 - Prompt {i+1}: {task_status} (重试 {retry_count}/{max_retries})")
                            continue
                    else:
                        print(f"[ERROR] 查询任务状态失败 - Prompt {i+1}: HTTP {task_response.status_code}")
                        break
                
                if retry_count >= max_retries:
                    print(f"[ERROR] 任务超时 - Prompt {i+1}")
                    
            else:
                error_info = response.json() if response.content else {"message": "Unknown error"}
                print(f"[ERROR] FLUX API调用失败 - Prompt {i+1}: {error_info.get('message', 'Unknown error')}")

        except Exception as e:
            print(f"[ERROR] FLUX图像生成异常 - Prompt {i+1}: {str(e)}")
        
        # 添加请求间隔，避免频率限制
        if i < len(prompts) - 1:
            time.sleep(1)

    return images

# ----------------------------------------------- #
def overlay_word_on_images(image_paths, word, position='bottom_center', opacity=180, font_size_ratio=0.07,
                           bg_bar=True, bg_opacity=120, bg_padding_ratio=0.012, corner_ratio=0.02,
                           center_bias_ratio=-0.06):
    """Overlay the exact vocabulary word onto images to avoid misspelling by T2I models.
    Args:
        image_paths (list[str]): paths to source images
        word (str): the exact word to render
        position (str): 'bottom_center' | 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right'
        opacity (int): 0-255 alpha of the text fill
        font_size_ratio (float): font size as ratio of min(image_width, image_height)
    Returns:
        list[str]: paths to the overlaid images (saved as new files with suffix '_text')
    """
    out_paths = []
    if not image_paths:
        return out_paths


    # Try to load a system font; fall back to default
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    def load_font(size):
        for fp in font_candidates:
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
        return ImageFont.load_default()

    for p in image_paths:
        try:
            # Ensure file handle is closed promptly to avoid Windows locks
            with Image.open(p) as base:
                img = base.convert("RGBA")
            w, h = img.size
            font_size = int(min(w, h) * font_size_ratio)
            font = load_font(font_size)

            # Create overlay layer
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # Measure text
            bbox = draw.textbbox((0, 0), word, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Determine position
            margin = int(0.03 * h)
            if position == 'top_left':
                x, y = margin, margin
            elif position == 'top_right':
                x, y = w - tw - margin, margin
            elif position == 'bottom_left':
                x, y = margin, h - th - margin
            elif position == 'bottom_right':
                x, y = w - tw - margin, h - th - margin
            else:  # bottom_center
                x, y = (w - tw) // 2, h - th - margin

            # Optional semi-transparent bar behind text for readability
            if bg_bar:
                pad = int(min(w, h) * bg_padding_ratio)
                # Shift the background bar slightly downward for readability
                shift = int(pad * 0.6)
                rx1 = x - pad
                ry1 = y - pad + shift
                rx2 = x + tw + pad
                ry2 = y + th + pad + shift
                rad = int(min(w, h) * corner_ratio)
                rect_fill = (0, 0, 0, bg_opacity)
                try:
                    draw.rounded_rectangle([(rx1, ry1), (rx2, ry2)], radius=rad, fill=rect_fill)
                except Exception:
                    draw.rectangle([(rx1, ry1), (rx2, ry2)], fill=rect_fill)

                # Compute bar center and place text by true center with a slight optical bias
                cx = (rx1 + rx2) / 2.0
                cy = (ry1 + ry2) / 2.0 + center_bias_ratio * th
                # Clamp cy to keep full text inside the bar bounds
                cy = max(ry1 + th / 2.0, min(cy, ry2 - th / 2.0))

                fill = (255, 255, 255, opacity)
                stroke_fill = (0, 0, 0, min(220, opacity + 40))
                stroke_w = max(1, int(font_size * 0.08))
                # Prefer stroke rendering with anchor='mm'; fallback to legacy outline if unsupported
                try:
                    draw.text((cx, cy), word, font=font, fill=fill, anchor='mm', stroke_width=stroke_w, stroke_fill=stroke_fill)
                except TypeError:
                    # Legacy Pillow: approximate with 4-direction outline
                    x_legacy = int(cx - tw / 2.0)
                    y_legacy = int(cy - th / 2.0)
                    outline = stroke_fill
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((x_legacy + dx, y_legacy + dy), word, font=font, fill=outline)
                    draw.text((x_legacy, y_legacy), word, font=font, fill=fill)
            else:
                # No bar: keep original position logic, but still use stroke when available
                cx = x + tw / 2.0
                cy = y + th / 2.0 + center_bias_ratio * th
                fill = (255, 255, 255, opacity)
                stroke_fill = (0, 0, 0, min(220, opacity + 40))
                stroke_w = max(1, int(font_size * 0.08))
                try:
                    draw.text((cx, cy), word, font=font, fill=fill, anchor='mm', stroke_width=stroke_w, stroke_fill=stroke_fill)
                except TypeError:
                    outline = stroke_fill
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((x + dx, y + dy), word, font=font, fill=outline)
                    draw.text((x, y), word, font=font, fill=fill)

            composed = Image.alpha_composite(img, overlay).convert("RGB")
            out_path = p[:-4] + "_text.png" if p.lower().endswith('.png') else p + "_text.png"
            composed.save(out_path)
            out_paths.append(out_path)
        except Exception as e:
            print(f"[WARN] overlay failed for {p}: {e}")
            out_paths.append(p)
    
    return out_paths

def query_subject(subject, vocab, grammar, skill):
    # 若subject为字符串，用此得到info dict
    # 否则subject为dict，用table和uid信息查询proficiency
    if subject is None: return None
    elif isinstance(subject, str):
        match_uids = vocab[vocab['word'] == subject]['uid'].tolist()
        if not match_uids: # 不在metadata中的词汇
            return {
                'content': subject,
                'table': 'vocab',
                'uid': None,
                'proficiency': 0.5 # 未知词汇取默认掌握度
            }
        uid = match_uids[0] # 选择一个作为代表
        match_skills = skill[(skill['table'] == 'vocab') & (skill['uid'].isin(match_uids))]
        if match_skills.empty: # 此词还没学过
            proficiency = 0
        else: # 一个词可能以多个义项出现，取平均掌握度
            proficiency = match_skills['proficiency'].mean()
        
        return {
            'content': subject, 
            'table': 'vocab',
            'uid': uid,
            'proficiency': proficiency
        }
    elif isinstance(subject, dict):
        # 安全处理：允许uid为None（词汇不在metadata中）
        ret = subject.copy()
        # 若未提供table，默认按vocab处理
        if 'table' not in ret:
            ret['table'] = 'vocab'
        # uid为None时，直接保留调用方提供的content与proficiency
        if ret.get('uid') is None:
            ret['content'] = ret.get('content', '')
            ret['proficiency'] = ret.get('proficiency', 0.5)
            return ret

        # uid有效：从metadata读取更规范的内容字段
        item = get_metadata_item(ret, vocab, grammar)
        # Avoid ambiguous truth value on pandas.Series; normalize to dict if possible
        try:
            if hasattr(item, 'to_dict'):
                item = item.to_dict()
        except Exception:
            pass
        try:
            if ret['table'] == 'vocab' and ('word' in item):
                ret['content'] = item['word']
            elif ret['table'] != 'vocab' and ('grammar' in item):
                ret['content'] = item['grammar']
        except Exception:
            pass
        # 同步熟练度（若skill表有记录）
        match_skill = skill[(skill['table'] == ret['table']) & (skill['uid'] == ret['uid'])]
        if not match_skill.empty:
            ret['proficiency'] = match_skill['proficiency'].iloc[0]
        else:
            # 保留调用方提供的熟练度；若未提供则回退为0
            ret['proficiency'] = ret.get('proficiency', 0)
        return ret
    
def explaining_style(level):
    # 视CEFR级别决定知识的基本解释风格（健壮处理：level 可能是 pandas.Series 或非字符串）
    try:
        # 若为序列/Series，取第一个标量
        if hasattr(level, "__iter__") and not isinstance(level, (str, bytes)):
            # 处理 pandas.Series / list / tuple
            try:
                level = next(iter(level))
            except Exception:
                pass
        lv = str(level) if level is not None else ''
    except Exception:
        lv = ''

    order = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    rank = order.index(lv) if lv in order else None

    if rank == 0:  # A1
        return 'concrete association'
    elif rank is not None and rank < order.index('B2'):
        return 'analogical'
    else:
        return 'metaphorical'

def explaining_complexity(proficiency):
    # 视知识掌握度决定解释的句式、词汇复杂度（健壮处理：proficiency 可能是 pandas.Series 或 numpy 标量）
    try:
        if hasattr(proficiency, "__iter__") and not isinstance(proficiency, (str, bytes)):
            try:
                proficiency = float(next(iter(proficiency)))
            except Exception:
                proficiency = float(proficiency.iloc[0]) if hasattr(proficiency, 'iloc') else float(proficiency)
        else:
            proficiency = float(proficiency)
    except Exception:
        proficiency = 0.0

    if proficiency < 0.3:
        return 'Express with brief sentences and simple vocabulary. '
    elif proficiency < 0.6:
        return 'Express with daily expressions, using more examples. '
    else:
        return 'Express in intermediate language complexity, using analogies, comparisons or metaphors. '

def _meta_norm_str(meta, key, default=""):
    """Safely extract a scalar string from meta (dict or pandas.Series).
    Treat None/NaN/empty as default. Avoid pandas Series truthiness.
    """
    try:
        val = meta.get(key, default) if hasattr(meta, 'get') else (meta[key] if key in meta else default)
    except Exception:
        try:
            val = meta[key]
        except Exception:
            val = default
    # Normalize NaN/None
    try:
        import math
        if val is None:
            return default
        # numpy.nan or float('nan') handling
        if isinstance(val, float) and math.isnan(val):
            return default
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() == 'nan':
        return default
    return s

def format_knowledge(subject, meta=dict()):
    # 将被解释对象进行文本格式化，以嵌入prompt
    # meta是dict或pd.Series，若为空则表示被解释对象不存在于metadata中
    if subject['table'] == 'vocab':
        typ = _meta_norm_str(meta, 'type')
        topic = _meta_norm_str(meta, 'topic')
        level = _meta_norm_str(meta, 'level')
        knowledge = (
            f"word / expression: {subject['content']}\n"
            +(f"part of speech: {typ}\n" if typ else '')
            +(f"related topic: {topic}\n" if topic else '')
            +(f"recommended explaining style: {explaining_style(level)}\n" if level else '')
        )
    else:
        typ = _meta_norm_str(meta, 'type')
        example = _meta_norm_str(meta, 'example')
        level = _meta_norm_str(meta, 'level')
        knowledge = (
            f"grammar point: {subject['content']}\n"
           +(f"grammar category: {typ}\n" if typ else '')
           +(f"examples: \n{example}\n" if example else '')
           +(f"recommended explaining style: {explaining_style(level)}\n" if level else '')
        )
    return knowledge

def init_subdialogue_context(root_context, responser, vocab, grammar, world, skill, subject=None, **kwargs):
    # NOTE: subject为被解释话题
    # - 从会话文字点击传送的单词，subject为单词的字符串
    # - 从特殊高亮的知识点传送的单词，subject为一个info dict，包含table + uid + proficiency

    HISTORY_ATTENTION_LEN = 10 # 最多关注的历史消息条数（不包括prompt）
    pure_context = get_context_without_prompt(root_context)
    role_name = {
        'assistant': world['meta']['system_role'],
        'user': world['meta']['user_role']
    }
    attended_history = [role_name[item['role']] + ": " + item['content'] for item in pure_context[-HISTORY_ATTENTION_LEN:]] # 最多提取x条
    attended_history = "\n".join(attended_history)

    if attended_history:
        messages = [{
            "role": "system",
            "content": "Draw a summary about the topic and main content of the following conversation. "
                    "Give your summary as comma-separated list of keywords or phrases (no more than 30 words in total). \n"
                    + attended_history + "\n"
        }]

        background = get_response(**responser, context=messages)
    else:
        background = ""
    #print("background: ", background)
    print("SUBJECT:", subject)
    subject = query_subject(subject, vocab, grammar, skill)

    if subject is None:
        knowledge = ''
    elif subject['table'] == 'vocab':
        if subject['uid'] is None:
            knowledge = (
                f"word / expression: {subject['content']}"
            )
        else:
            sel = vocab[vocab['uid'] == subject['uid']]
            meta = sel.iloc[0] if not sel.empty else {}
            knowledge = format_knowledge(subject, meta)
    else:
        sel = grammar[grammar['uid'] == subject['uid']]
        meta = sel.iloc[0] if not sel.empty else {}
        knowledge = format_knowledge(subject, meta)



    goal = f"You will explain the following knowledge as {role_name['user']} requests:\n{knowledge}\n" if knowledge else ""
    complexity = explaining_complexity(subject['proficiency']) if knowledge else ""

    context = [
        {
            "role": "system",
            "content": f"You are {role_name['assistant']}, who is the language mentor of "
                       f"{role_name['user']} in a virtual world. "
                       + goal + complexity + 
                       f"Always respond in {world['meta']['preferences']['target_language']} "
                       f"and the response must not be too long (respond like chatting). "
                       f"GIVE PURE TEXT WITHOUT ANY FORMATS. "
                       +
                       (("The following summary of your previous conversation "
                        f"may help you better illustrate the knowledge:\n{background}\n") if background else "")
                       +
                       f"Now tell {role_name['user']} to feel free to chat or ask about the knowledge "
                       f"in {world['meta']['preferences']['target_language']} (no more than 15 words). "
        },
    ]

    context.append({
        "role": "assistant",
        "content": get_response(**responser, context=context)
        })
    return context


def generate_case_example(subject, meta, world, context, responser):
    attended_context = format_attended_context(context, world, att_len=6)
    prompt_context = f"Its idea can be related to the recent conversation:\n{attended_context}"
    prompt = (
        f"Generate an example sentence in {world['meta']['preferences']['target_language']} which makes use of the following knowledge:\n{format_knowledge(subject, meta)}" # 包含基于CEFR级别的解释风格推荐
        f"{explaining_complexity(subject['proficiency'])}" # 基于熟练度的句式复杂度推荐
        +prompt_context
        +"OUTPUT THE EXAMPLE SENTENCE WITHOUT ANY EXTRACT TEXT. "
    )
    return get_response(context=[{"role": "user", "content": prompt}], **responser)

def generate_image_prompts(subject, meta, example, count, responser):
    # Determine the single-word text (only for vocab). For grammar, prefer no text in image.
    target_word = subject.get('content') if subject.get('table') == 'vocab' else None
    knowledge_txt = format_knowledge(subject, meta)
    # 选择主体类型优先级
    agent_priority = determine_agent_types(subject, meta if isinstance(meta, dict) else meta.to_dict() if hasattr(meta, 'to_dict') else {}, example)
    agent_hint = (
        "Primary agent should be HUMAN (include clear human presence keywords like person, face, portrait, crowd)."
        if agent_priority and agent_priority[0] == 'human' else
        "Primary agent should be ANIMAL (specify species if applicable, include animal keywords)."
        if agent_priority and agent_priority[0] == 'animal' else
        "Primary agent should be OBJECT (focus on objects and their usage, avoid people unless necessary)."
    )

    # 先尝试 Planner → Compose 路径
    try:
        plan = plan_visual_semantics(subject, meta if isinstance(meta, dict) else meta.to_dict() if hasattr(meta, 'to_dict') else {}, example, responser)
        candidates = compose_from_plan(plan, k=max(count, 4), agent_priority=agent_priority)
        cleaned_cands = []
        for ln in candidates:
            q = ln.replace("```", "").replace('"', "").replace("'", "").strip()
            import re
            q = re.sub(r"\b(text|word|letters?|engraved|printed|label|caption|subtitle|writing|handwriting)\b[^,]*,?\s*", "", q, flags=re.IGNORECASE)
            q = re.sub(r",\s*,+", ", ", q).strip().strip(',').strip()
            cleaned_cands.append(q)
        # 主/次通道打分
        scored = [(score_prompt_for_agent(q, agent_priority[0] if agent_priority else 'human'), q) for q in cleaned_cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [q for _, q in scored[:count]]
        if not selected and cleaned_cands:
            selected = cleaned_cands[:count]
        if selected:
            return selected
    except Exception as e:
        print(f"[WARN] planner failed, fallback to direct LLM prompts: {e}")

    # 我们对LLM多采样，随后用启发式选优（回退路径）
    samples = max(count, 3)
    prompt = (
        f"Design prompts for generating {count} image(s) that visualize the sentence: {example}\n"
        f"Focus on illustrating the role of the following knowledge in the scene:\n{knowledge_txt}\n"
        # Text rendering constraints (strictly no text inside images)
        + ("Do not render any text, letters, words, captions, labels, or handwriting inside the image. Use pure visual metaphors only. ")
        # Composition and abstraction guidance
        + (
            "For abstract concepts, prioritize logical comparisons within the same category (e.g., time concepts: 'yesterday' vs 'today' vs 'tomorrow' on a calendar), use timelines, calendars, or consistent visual motifs to contrast meanings. "
            "For concrete words, depict clear objects/scenes with minimal clutter, using arrows, highlights, or simple icons. "
        )
        # Agent guidance
        + (agent_hint + " Ensure the agent type is visually central; if ambiguous (e.g., adjectives), prefer the primary agent but you may include subtle cues for the secondary. ")
        # Output format
        + (
            f"Output exactly {samples} line(s). Each line is a single image prompt as a concise, comma-separated list of keywords and short visual descriptors (no full sentences). "
            "Do NOT include markdown, quotes, backticks, or explanations in the output; only the prompts. "
        )
    )
    raw = get_response(context=[{"role": "user", "content": prompt}], **responser).strip()
    lines = [ln.strip() for ln in raw.split('\n') if ln.strip()]
    # enforce exact sample count and sanitize common formatting artifacts
    lines = lines[:samples]
    cleaned = []
    for ln in lines:
        q = ln.replace("```", "")
        q = q.replace('"', "").replace("'", "")
        q = q.strip()
        # remove any lingering text-related directives/tokens
        import re
        q = re.sub(r"\b(text|word|letters?|engraved|printed|label|caption|subtitle|writing|handwriting)\b[^,]*,?\s*", "", q, flags=re.IGNORECASE)
        q = re.sub(r",\s*,+", ", ", q).strip().strip(',').strip()
        cleaned.append(q)
    # 基于主体类型对候选进行打分并选Top-1..count
    scored = []
    primary = agent_priority[0] if agent_priority else 'human'
    secondary = agent_priority[1] if agent_priority and len(agent_priority) > 1 else None
    for q in cleaned:
        scored.append((score_prompt_for_agent(q, primary), q))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [q for _, q in scored[:count]]
    # 若主通道不理想（得分过低）且有次通道，尝试用次通道重打分
    if secondary and (not selected or (scored and scored[0][0] < 0.2)):
        scored2 = [(score_prompt_for_agent(q, secondary), q) for q in cleaned]
        scored2.sort(key=lambda x: x[0], reverse=True)
        selected = [q for _, q in scored2[:count]]
    return selected

def extract_topic_keywords(context, world, responser):
    pure_context = format_attended_context(context, world, att_len=None)
    
    prompt = f"""
    Extract 3 to 5 topic keywords from the following conversation (output in comma-seperated list):\n{pure_context}
    """
    keywords = get_response(context=[{"role": "user", "content": prompt}], **responser)
    return [kw.strip() for kw in keywords.split(',')]


def interactive_interpreter(context, responser, **kwargs):
    # 交互解释（目前暂不考虑利用metadata和archive中的信息）

    # NOTE: 子会话不必每次采取提取话题词汇的方式，会造成会话内上下文不连续
    #       掌握度机制嵌入到一开头的prompt中

    context = get_context_pure_dialogue(context)

    explanation = get_response(**responser, context=context)
    
    return {
        "content": explanation,
        "info": {
            "type": "interactive_explanation",
        }
    }

def instance_interpreter(context, responser, scenarios, vocab, grammar, vocab_emb, 
                         world, skill, progress, flux_cfg, subject=None, overlay_text=True, output_dir: str = "generated_images", **kwargs):
    # 案例解释
    # NOTE: responser为client和model的字典，可以直接用get_response(context, **responser, **kwargs)获取百炼LLM的回复，更便捷
    # subject: str or dict
    #   若为str则是被解释的词汇文本
    #   为dict则是更具体信息，包含
    #   - table
    #   - uid(可能为None，表示词汇不在现有的metadata中)
    #   - content(词汇的word字段或语法点的grammar字段),
    #   - proficiency
    
    subject = query_subject(subject, vocab, grammar, skill) # 统一为dict格式

    meta = get_metadata_item(subject, vocab, grammar)

    target_lang = world['meta']['preferences']['target_language']
    # 获取元数据和掌握度
        
    # 生成例句
    example = generate_case_example(subject, meta, world, context, responser)
        
    # 生成图片提示词
    image_prompts = generate_image_prompts(subject, meta, example, DEFAULT_IMAGE_COUNT, responser)
        
    # 调用文生图模型生成图片
    newline = '\n'
    print(f"[INFO] generated image prompts:\n{newline.join(image_prompts)}")
    # if flux_cfg: # FLUX
    image_paths = generate_images_flux(image_prompts, flux_cfg, target_lang, output_dir=output_dir)
    
    """else: # SD
        image_paths = generate_images(image_prompts, sd_pipeline, output_dir=output_dir)"""
    
    # ---------- CLIP reranking + quality gates ----------
    diagnostics = {
        "rerank": [],
        "goals": [],
        "quality": [],
        "plan_used": None,
        "plan_fields": {},  # sense/required_cues/forbidden_cues/composition (if available)
        "agent_priority": [],
        "primary_agent": None,
        "clip_fallback_overall": None,
    }
    # Derive agent priority and plan for goals (best-effort)
    meta_dict = meta if isinstance(meta, dict) else meta.to_dict() if hasattr(meta, 'to_dict') else {}
    agent_priority = determine_agent_types(subject, meta_dict, example)
    primary_agent = agent_priority[0] if agent_priority else 'human'
    diagnostics["agent_priority"] = list(agent_priority) if agent_priority else []
    diagnostics["primary_agent"] = primary_agent
    plan_for_goals = None
    try:
        plan_for_goals = plan_visual_semantics(subject, meta_dict, example, responser)
        diagnostics["plan_used"] = True
        # Extract key planner fields for diagnostics
        diagnostics["plan_fields"] = {
            "sense": (plan_for_goals.get("sense") or ""),
            "required_cues": plan_for_goals.get("required_cues", []) or [],
            "forbidden_cues": plan_for_goals.get("forbidden_cues", []) or [],
            "composition": plan_for_goals.get("composition", []) or [],
        }
    except Exception:
        diagnostics["plan_used"] = False
    goals = build_text_goals_from_plan_or_context(subject, meta_dict, example, plan_for_goals)
    diagnostics["goals"] = goals

    # Keep a copy of all generated paths for potential cleanup after selection
    _all_generated_paths = list(image_paths or [])

    # Score and gate each generated image
    scored = []
    gated = []
    for p in (image_paths or []):
        try:
            # No need to open image; operate on paths to avoid file locks
            pass
        except Exception as e:
            diagnostics["quality"].append({"path": p, "error": str(e)})
            continue
        # Use image path for CLIP-based functions
        gates = quality_gates_clip(p, primary_agent)
        score = clip_score_image_against_goals(p, goals)
        diagnostics["quality"].append({"path": p, **gates})
        diagnostics["rerank"].append({"path": p, "clip_score": score})
        scored.append((score, p))
        gated.append((gates.get("text_free", True) and gates.get("agent_present", True), p))

    # Prefer top score among those passing gates; else fallback to top score overall
    if scored:
        passing = [sp for g, sp in gated if g]
        if passing:
            scored_pass = [sp for sp in scored if sp[1] in passing]
            scored_pass.sort(key=lambda x: x[0], reverse=True)
            best_path = scored_pass[0][1]
        else:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_path = scored[0][1]
        # Keep only the best image for downstream saving/reporting
        image_paths = [best_path]
        # Proactively remove non-selected images to avoid clutter
        try:
            import os
            for other in _all_generated_paths:
                if other != best_path and os.path.isfile(other):
                    try:
                        os.remove(other)
                    except Exception:
                        pass
        except Exception:
            pass

    # Aggregate CLIP fallback marker across images
    try:
        diagnostics["clip_fallback_overall"] = any(
            bool(q.get("clip_fallback")) for q in diagnostics.get("quality", []) if isinstance(q, dict)
        ) if diagnostics.get("quality") else None
    except Exception:
        diagnostics["clip_fallback_overall"] = None

    # Optional: overlay exact vocabulary word (default disabled to keep images text-free)
    if overlay_text and image_paths and subject and subject.get('table') == 'vocab':
        try:
            # Keep original best for now to create overlaid variant
            original_best = image_paths[0]
            image_paths = overlay_word_on_images(image_paths, subject.get('content', ''), position='bottom_center')
            # Remove the original non-overlaid file to avoid ambiguity in generated_images
            try:
                import os
                if image_paths and original_best != image_paths[0] and os.path.isfile(original_best):
                    os.remove(original_best)
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] overlay pipeline failed: {e}")
            
    # 构建响应消息：始终返回 prompts，若生成失败则提供 prompts 便于检查
    return {
        "content": image_paths if image_paths else image_prompts,
        "info": {
            "type": "instance_explanation",
            "subject": subject,
            "example": example,
            "image_count": len(image_paths),
            "image_paths": image_paths,
            "image_prompts": image_prompts,
            "diagnostics": diagnostics,
        }
    }


