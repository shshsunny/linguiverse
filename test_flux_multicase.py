import os
import shutil
import json
from datetime import datetime
from pprint import pprint

from openai import OpenAI
from utils import read_apikey
from modules.subdialogue import init_subdialogue_context, instance_interpreter, init_flux_pipeline
from modules.data_manager import DataManager


def _move_to_flux_test(paths):
    """Deprecated after output_dir alignment: kept for backward compatibility."""
    return paths or []


def run_case(case_id: str, language: str, subject: dict, extra_context: list, responser: dict):
    dm = DataManager(language, debugging=False)
    world = dm.get_archive()['world']

    ctx = init_subdialogue_context([], responser, subject=subject, **dm.get_metadata(), **dm.get_archive())
    if extra_context:
        ctx.extend(extra_context)

    flux_cfg = init_flux_pipeline(api_key=read_apikey("config/api.key"), model_name="flux-schnell")
    result = instance_interpreter(
        ctx, responser, subject=subject, flux_cfg=flux_cfg, overlay_text=True, output_dir="flux_test",
        **dm.get_metadata(), **dm.get_archive()
    )
    info = result.get('info', {})
    prompts = info.get('image_prompts', []) or []
    paths = info.get('image_paths', []) or []
    moved_paths = _move_to_flux_test(paths)

    report = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'case_id': case_id,
        'language': language,
        'subject': info.get('subject', subject),
        'example': info.get('example', ''),
        'image_prompts': prompts,
        'image_paths': moved_paths,
        'diagnostics': info.get('diagnostics', {}),
        'content_field': result.get('content'),
    }
    out_path = os.path.join('flux_test', f"report_{case_id}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out_path}")
    return result


def main():
    apikey_path = "config/api.key"
    responser = {
        'model': 'qwen-plus-latest',
        'client': OpenAI(
            api_key=read_apikey(apikey_path),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    }

    cases = [ 
        # FR baseline known word (vocab in metadata)
        ("fr_timide", 'french', {'table': 'vocab', 'uid': 7395, 'content': 'timide', 'proficiency': 0.5}, [
            {"role": "user", "content": "Qu'est-ce que 'timide' veut dire ?"}
        ]),
        # EN out-of-metadata common object (object-centric)
        ("en_umbrella", 'english', {'table': 'vocab', 'uid': None, 'content': 'umbrella', 'proficiency': 0.4}, [
            {"role": "user", "content": "What is an umbrella used for?"}
        ]),
        # EN animal word (animal-centric)
        ("en_cow", 'english', {'table': 'vocab', 'uid': None, 'content': 'cow', 'proficiency': 0.4}, [
            {"role": "user", "content": "Can you show me what a cow looks like?"}
        ]),
        # EN abstract concept (human-centric context likely)
        ("en_freedom", 'english', {'table': 'vocab', 'uid': None, 'content': 'freedom', 'proficiency': 0.6}, [
            {"role": "user", "content": "Explain the idea of freedom in a simple way."}
        ]),
        # FR everyday object (out-of-metadata)
        ("fr_lampe", 'french', {'table': 'vocab', 'uid': None, 'content': 'lampe', 'proficiency': 0.5}, [
            {"role": "user", "content": "Qu'est-ce qu'une lampe ?"}
        ]),
        # EN multi-word phrase (tests tokenizer/prompting)
        ("en_coffee_shop", 'english', {'table': 'vocab', 'uid': None, 'content': 'coffee shop', 'proficiency': 0.5}, [
            {"role": "user", "content": "What is a coffee shop?"}
        ]),
        # --- Harder abstract/ambiguous concepts ---
        ("en_melancholy", 'english', {'table': 'vocab', 'uid': None, 'content': 'melancholy', 'proficiency': 0.6}, [
            {"role": "user", "content": "What does melancholy feel like? Show me."}
        ]),
        ("en_entropy", 'english', {'table': 'vocab', 'uid': None, 'content': 'entropy', 'proficiency': 0.6}, [
            {"role": "user", "content": "Explain entropy visually without using equations."}
        ]),
        ("en_nothingness", 'english', {'table': 'vocab', 'uid': None, 'content': 'nothingness', 'proficiency': 0.6}, [
            {"role": "user", "content": "Depict the idea of nothingness without using text."}
        ]),
        ("en_paradox", 'english', {'table': 'vocab', 'uid': None, 'content': 'paradox', 'proficiency': 0.6}, [
            {"role": "user", "content": "Visualize a paradox in a single scene."}
        ]),
        ("en_liminality", 'english', {'table': 'vocab', 'uid': None, 'content': 'liminality', 'proficiency': 0.6}, [
            {"role": "user", "content": "What is liminality? Give a visual impression."}
        ]),
        ("en_serendipity", 'english', {'table': 'vocab', 'uid': None, 'content': 'serendipity', 'proficiency': 0.6}, [
            {"role": "user", "content": "Illustrate serendipity without using any words."}
        ]),
        ("en_superposition", 'english', {'table': 'vocab', 'uid': None, 'content': 'superposition', 'proficiency': 0.6}, [
            {"role": "user", "content": "Show the idea of quantum superposition metaphorically."}
        ]),
        ("en_zeitgeist", 'english', {'table': 'vocab', 'uid': None, 'content': 'zeitgeist', 'proficiency': 0.6}, [
            {"role": "user", "content": "Depict the zeitgeist of a generation in one image."}
        ]),
        ("fr_absurde", 'french', {'table': 'vocab', 'uid': None, 'content': 'absurde', 'proficiency': 0.6}, [
            {"role": "user", "content": "Montre l'absurde sans utiliser de texte."}
        ]),
        ("fr_liberté", 'french', {'table': 'vocab', 'uid': None, 'content': 'liberté', 'proficiency': 0.6}, [
            {"role": "user", "content": "Représente l'idée de la liberté en une seule scène."}
        ]),
        ("en_butterfly_effect", 'english', {'table': 'vocab', 'uid': None, 'content': 'butterfly effect', 'proficiency': 0.6}, [
            {"role": "user", "content": "Visualize the butterfly effect without formulas or text."}
        ]),
        ("en_sublime", 'english', {'table': 'vocab', 'uid': None, 'content': 'the sublime', 'proficiency': 0.6}, [
            {"role": "user", "content": "Convey the feeling of the sublime visually."}
        ]),
        ("en_ambiguity", 'english', {'table': 'vocab', 'uid': None, 'content': 'ambiguity', 'proficiency': 0.6}, [
            {"role": "user", "content": "Create an image that intentionally remains ambiguous."}
        ]),
    ]

    os.makedirs('flux_test', exist_ok=True)
    for case_id, lang, subject, extra_ctx in cases:
        try:
            print(f"\n[CASE] {case_id} ({lang}) -> {subject['content']}")
            run_case(case_id, lang, subject, extra_ctx, responser)
        except Exception as e:
            print(f"[ERROR] case {case_id}: {e}")


if __name__ == '__main__':
    main()
