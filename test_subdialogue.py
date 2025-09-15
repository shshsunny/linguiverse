from modules.subdialogue import init_subdialogue_context, instance_interpreter, init_flux_pipeline
from modules.data_manager import DataManager
from test import get_test
from utils import read_apikey
import torch
from openai import OpenAI
from pprint import pprint
import json
from datetime import datetime
import os
import shutil

def test_init(responser):
    dm = DataManager('french', debugging=False)
    world = dm.get_archive()['world']
    context = [
        {
            "role": "system",
            "content": "Tu es Oracle Lys, une conscience végétale. Tu guides Écho, une gardienne sensible au souffle de la cité flottante Néphélis."
        },
        {
            "role": "user",
            "content": "Oracle, j’ai senti une vibration étrange près du dôme Est."
        },
        {
            "role": "assistant",
            "content": "Le dôme Est ne respire plus en rythme. Ce n’est pas normal."
        },
        {
            "role": "user",
            "content": "Les feuilles vibrent trop vite. Même le silence est tendu."
        },
        {
            "role": "assistant",
            "content": "Les signes sont clairs. Quelque chose se déplace sous les racines."
        },
        {
            "role": "user",
            "content": "Dois-je descendre sous le corridor bleu ?"
        },
        {
            "role": "assistant",
            "content": "Oui. Mais prends ton manteau d’algues. L’air y est instable."
        },
        {
            "role": "user",
            "content": "Les lucioles-sondes sont déjà parties. Elles brillent faiblement."
        },
        {
            "role": "assistant",
            "content": "Suis leur lumière. Elles te mèneront à la faille."
        },
        {
            "role": "user",
            "content": "Je sens une ancienne peur. Elle flotte dans l’air."
        },
        {
            "role": "assistant",
            "content": "Cette peur n’est pas la tienne. C’est un souvenir de Néphélis."
        },
        {
            "role": "user",
            "content": "Je vais marcher lentement. Je resterai à l’écoute."
        },
        {
            "role": "assistant",
            "content": "Alors le Souffle sera avec toi. Parle-moi dès que tu vois l’arche fracturée."
        }
    ]
    #subdialogue_context = init_subdialogue_context(context, responser, world, keyword='fracturée.')
    #pprint(subdialogue_context)

def test_instance_interpreter(responser):

    dm = DataManager('french', debugging=False)
    world = dm.get_archive()['world']
    subject = {
        'table': 'vocab',
        'uid': 7395,
        'content': 'timide',
        'proficiency': 0.5,
    }
    context = init_subdialogue_context([], responser, subject=subject, **dm.get_metadata(), **dm.get_archive())
    pprint(context)
    context.extend([
        {
            "role": "user",
            "content": "Oracle Lys, qu'est-ce que 'timide' veut dire ? Je l'ai lu dans le Codex, mais je ne comprends pas."
        },
        {
            "role": "system",
            "content": "Écho, 'timide' décrit une personne qui a peur de parler avec les autres. Elle préfère rester tranquille."
        },
        {
            "role": "user",
            "content": "On peut dire ça pour un animal ? Comme les oiseaux du jardin ?"
        },
        {
            "role": "system",
            "content": "Oui. Un animal timide fuit quand on s'approche. Les oiseaux de ce jardin sont timides : ils s'envolent vite."
        },
        {
            "role": "user",
            "content": "Et si je ne veux pas parler en public, est-ce que c'est être timide ?"
        },
        {
            "role": "system",
            "content": "Oui. Si tu as peur de parler devant beaucoup de gens, tu es timide dans cette situation. C'est être réservé."
        }
    ])

    # Initialize FLUX config (DashScope) and pass to instance_interpreter
    flux_cfg = init_flux_pipeline(api_key=read_apikey("config/api.key"), model_name="flux-schnell")
    image_message = instance_interpreter(
        context, responser, subject=subject, flux_cfg=flux_cfg, overlay_text=True, **dm.get_metadata(), **dm.get_archive()
    )
    # Print for manual inspection
    pprint(image_message)

    # Write all evaluation info to a txt file for review
    info = image_message.get('info', {})
    prompts = info.get('image_prompts', []) or []
    paths = info.get('image_paths', []) or []
    subj = info.get('subject', {}) or {}
    example = info.get('example', '')

    # Move generated images into 'flux_test' folder (if any)
    if paths:
        os.makedirs('flux_test', exist_ok=True)
        moved_paths = []
        for src in paths:
            base = os.path.basename(src)
            dest = os.path.join('flux_test', base)
            # ensure unique filename
            if os.path.exists(dest):
                name, ext = os.path.splitext(base)
                k = 1
                while os.path.exists(os.path.join('flux_test', f"{name}_{k}{ext}")):
                    k += 1
                dest = os.path.join('flux_test', f"{name}_{k}{ext}")
            try:
                shutil.move(src, dest)
                moved_paths.append(dest)
            except Exception:
                # if move fails (e.g., cross-device), try copy then remove
                try:
                    shutil.copy2(src, dest)
                    os.remove(src)
                    moved_paths.append(dest)
                except Exception:
                    moved_paths.append(src)
        paths = moved_paths

    # Lightweight format checks summary
    checks = []
    for i, p in enumerate(prompts):
        checks.append({
            'index': i,
            'no_newline': ('\n' not in p),
            'no_codefence': ('```' not in p),
            'no_quotes': ('"' not in p and "'" not in p),
            'is_single_line': (p.count('\n') == 0),
            'raw': p,
        })

    report = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'subject': subj,
        'example': example,
        'image_prompts': prompts,
        'image_paths': paths,
        'prompt_checks': checks,
        'content_field': image_message.get('content'),
    }
    with open('test_subdialogue.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))

    # Basic assertions to ensure we always receive prompts even if images are not generated
    assert isinstance(image_message, dict)
    assert 'info' in image_message and isinstance(image_message['info'], dict)
    info = image_message['info']
    # Should include prompts and paths
    assert 'image_prompts' in info
    prompts = info['image_prompts']
    assert isinstance(prompts, list) and len(prompts) >= 1
    # Should not crash if images are not generated (paths can be empty)
    assert 'image_paths' in info
    assert isinstance(info['image_paths'], list)
    # Subject metadata should be echoed back
    assert info.get('subject', {}).get('content') == subject['content']
    # Light format checks: prompts should be concise, no markdown quotes
    for p in prompts:
        assert '\n' not in p
        assert '```' not in p
        assert '"' not in p and "'" not in p  # no quotes preferred in final prompt text


if __name__ == '__main__':
    apikey_path = "config/api.key"
    responser = {
    'model': 'qwen-plus-latest',
    'client': OpenAI(
        api_key=read_apikey(apikey_path),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    }
    #test_init(responser)
    test_instance_interpreter(responser)