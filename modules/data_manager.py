import os
import pandas as pd
import pickle  # 或者 json
import json
from pathlib import Path
import time
import torch
from utils import *
import fnmatch
import numpy as np  

# ==== CASCADE PATCH: added for JSON conversions ====

# ==== CASCADE PATCH START: safe JSON default converter ====
def _json_default(obj):
    """Safely convert numpy/pandas scalars/lists to JSON-serializable types."""
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (_pd.Timestamp,)):
            return obj.isoformat()
        if hasattr(obj, 'item'):
            try:
                return obj.item()
            except Exception:
                pass
    except Exception:
        pass
    # Fallback: string representation
    try:
        return str(obj)
    except Exception:
        return None
# ==== CASCADE PATCH END: safe JSON default converter ====

class DataManager:
    def __init__(self, language: str, root_dir: str = ".", metadata_dir="metadata", archive_dir="archives", history_dir="history", debugging=False, debug_scene=5):
        """
        初始化 DataManager 实例

        :param language: 目标语言（如 'english'）
        :param root_dir: 项目根目录
        :param metadata_dir: metadata 子目录名
        :param archive_dir: archives 子目录名
        """
        self.language = language.lower()
        self.root_dir = Path(root_dir)
        self.metadata_path = self.root_dir / metadata_dir / self.language
        self.archive_path = self.root_dir / archive_dir
        self.history_path = self.root_dir / history_dir

        # 初始化容器

        # metadata
        self.vocab = None
        self.grammar = None
        self.scenarios = None

        # archives
        self.world = None
        self.skill = None
        self.progress = None

        # 自动加载数据
        self._load_metadata()
        if debugging:
            self.reset_archives(scene_uid=debug_scene) # 将整个存档初始化为从某个场景开始学习的状态，并且世界存档有特定内容
        else:
            self._load_archives()

        self.status_cache = dict() # 缓存当前场景不常变更的状态，避免重复计算

    def _load_metadata(self):
        # 特别注意：不设置na_values，以避免将空字符串转为NaN
        self.vocab = pd.read_csv(self.metadata_path / "vocab.csv", na_values=[])
        self.grammar = pd.read_csv(self.metadata_path / "grammar.csv", na_values=[])
        self.scenarios = pd.read_csv(self.metadata_path / "scenarios.csv", na_values=[])
        self.vocab_emb = torch.load(self.metadata_path / "vocab_emb.pt") if (self.metadata_path / "vocab_emb.pt").exists() else None

    def _load_archives(self):
        """
        加载语言世界与学习记录存档
        """
        world_file = self.archive_path / f"{self.language}.world.archive"
        skill_file = self.archive_path / f"{self.language}.skill.archive"
        progress_file = self.archive_path / f"{self.language}.progress.archive"
        self.skill = pd.read_csv(skill_file, na_values=[])
        with open(world_file, "rb") as f:
            self.world = pickle.load(f)
        with open(progress_file, "rb") as f:
            self.progress = pickle.load(f)

    def save_archives(self):
        """
        保存当前的 world_archive 和 learning_archive
        """
        world_file = self.archive_path / f"{self.language}.world.archive"
        skill_file = self.archive_path / f"{self.language}.skill.archive"
        progress_file = self.archive_path / f"{self.language}.progress.archive" # 存储与用户进度有关的信息，及其他杂项

        #try:
        self.skill.to_csv(skill_file, index=False)
        with open(world_file, "wb") as f:
            pickle.dump(self.world, f)
        with open(progress_file, "wb") as f:
            pickle.dump(self.progress, f)
        return True
        
        #except Exception as e:
        #    print(f"History Saving Error: {str(e)}")
        #    return False
    
    def reset_archives(self, scene_uid): # initialize the archives for testing

        self.skill = pd.DataFrame(columns=['table', 'uid', 'good_cases', 'bad_cases', 'proficiency'])
        self._add_scene_to_skill(scene_uid)

        now = time.time()

        worlds = {
            'english': {
                'meta': {
                    'system_role': 'Miss Blackwood',
                    'user_role': 'Elinor',
                    'last_access_time': now,
                    'preferences': {
                        'target_language': 'english',
                        'source_language': 'chinese',
                        'world_genre': [
                            'real-world',
                            'school',
                            'contemporary life',
                            'learning and growth',
                            'slice of life'
                        ],
                        'world_prologue': (
                            "Elinor is a new student in a small town school. "
                            "She is learning English with her teacher, Miss Blackwood. "
                            "The classroom is quiet and friendly. "
                            "Elinor wants to improve her speaking and writing skills."
                        )
                    }
                },
                'static': [
                    ("Elinor is a 13-year-old student in a middle school.", now),
                    ("Miss Blackwood is Elinor’s English teacher.", now),
                    ("The school is in a quiet town near the hills.", now),
                    ("The classroom has ten students and a large window.", now),
                    ("Elinor sits near the window in the second row.", now),
                    ("Miss Blackwood always speaks slowly and clearly.", now)
                ],
                'dynamic': [
                    ("Elinor says 'Good morning' to Miss Blackwood.", now - 600),  # 10 min ago
                    ("Miss Blackwood asks Elinor a question.",       now - 540),  # 9 min ago
                    ("Elinor answers with a full sentence.",         now - 480),  # 8 min ago
                    ("Two students read a short story in class.",    now - 420),  # 7 min ago
                    ("Elinor writes three simple sentences.",        now - 360),  # 6 min ago
                    ("Miss Blackwood checks Elinor’s writing.",      now - 300),  # 5 min ago
                    ("Elinor learns five new words.",                now - 240),  # 4 min ago
                    ("The class listens to a short audio clip.",     now - 180),  # 3 min ago
                    ("Elinor writes down the new vocabulary.",       now - 120),  # 2 min ago
                    ("Miss Blackwood gives the class a short quiz.", now - 60),   # 1 min ago
                ]
            },
            'french': {
                'meta': {
                    'system_role': 'Oracle Lys',
                    'user_role': 'Écho',
                    'last_access_time': now,
                    'preferences': {
                        'target_language': 'french',
                        'source_language': 'english',
                        'world_genre': [
                            'science-fiction',
                            'écologie',
                            'mondes flottants',
                            'exploration intérieure',
                            'utopie fragile'
                        ],
                        'world_prologue': (
                            "Écho est une gardienne du Souffle dans la cité flottante de Néphélis, suspendue au-dessus des nuages. "
                            "Elle communique avec l’intelligence organique de la cité par l’intermédiaire de l’Oracle Lys, une entité mi-végétale mi-numérique. "
                            "Chaque jour, elle enregistre les pulsations de la ville, les rêves des habitants et les signaux météorologiques pour maintenir l’équilibre. "
                            "Mais depuis peu, Néphélis respire avec difficulté, et l’harmonie vacille."
                        )
                    }
                },
                'static': [
                    ("Écho est née dans la serre centrale, parmi les fougères lumineuses.", now),
                    ("L’Oracle Lys murmure dans les langues oubliées du vent.", now),
                    ("Néphélis est une cité flottante bio-symbiotique, suspendue à 12 000 mètres d’altitude.", now),
                    ("Chaque quartier de la ville est alimenté par un arbre-sphère énergétique.", now),
                    ("Écho porte un manteau tissé de fibres aériennes récoltées à la saison des brumes.", now),
                    ("Les enfants de la cité apprennent à lire les courants aériens avant même de marcher.", now)
                ],
                'dynamic': [
                    ("Écho capte une variation étrange dans la fréquence des racines du dôme sud.", now - 600),
                    ("L’Oracle Lys émet une lumière bleue instable en réponse.", now - 540),
                    ("Un silence inhabituel plane au-dessus du jardin des vents.", now - 480),
                    ("Écho déploie un essaim de lucioles-sondes vers la canopée haute.", now - 420),
                    ("Une feuille-mémoire tombe entre ses mains, marquée d’un glyphe inconnu.", now - 360),
                    ("L’Oracle projette un souvenir ancien : une chute, un exil, un pacte oublié.", now - 300),
                    ("Écho inscrit ces signes dans le Codex du Souffle.", now - 240),
                    ("Un grondement traverse les câbles racinaires de la cité.", now - 180),
                    ("Elle enclenche le chant stabilisateur dans les coursives chantantes.", now - 120),
                    ("L’Oracle Lys demande : « Écho, es-tu prête à descendre sous les nuages ? »", now - 60)
                ]
            }
        }

        self.world = worlds[self.language]

        self.progress = {
            'current_scene': scene_uid,  # 当前正在学习的场景
            'scene_completed': False,
        }

        def match_files(pattern, directory):
            """
            匹配指定目录下符合模式的文件
            """
            matching_files = []
            for filename in os.listdir(directory):
                if fnmatch.fnmatch(filename, pattern):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        matching_files.append(file_path)
            return matching_files
        
        matching_files = []

        history_path = str(self.history_path)
        audio_path = "generated_audios"
        image_path = "generated_images"

        # 删除当前target_lang所有scene的history文件
        # TODO: 修改生成的audios和images文件名，确保区分开属于不同archive组（target_lang）的资源
        matching_files.extend(match_files(f"{self.language}.scene=*.history.json", history_path))
        matching_files.extend(match_files(f"{self.language}-*.wav", audio_path))             
        matching_files.extend(match_files(f"{self.language}-*.png", image_path))
        
        if not matching_files:
            print(f"No outdated data files to delete")
            return
        
        # 遍历并删除找到的文件
        for file_path in matching_files:
            try:
                os.remove(file_path)
                #print(f"已删除文件: {file_path}")
            except OSError as e:
                print(f"failed to remove {file_path}: {e}")


        self.save_archives()
    
    def _add_scene_to_skill(self, scene_uid):
        # 将场景元数据添加到语言能力存档，初始化用户数据
        scene_vocab = self.vocab[self.vocab["scene_uid"] == scene_uid]
        scene_grammar = self.grammar[self.grammar["scene_uid"] == scene_uid]

        # 创建新 skill 条目的列表
        new_records = []

        
        for _, row in scene_vocab.iterrows():
            uid = row["uid"]
            if not ((self.skill["table"] == "vocab") & (self.skill["uid"] == uid)).any():
                new_records.append({
                    "table": "vocab",
                    "uid": uid,
                    "good_cases": "",
                    "bad_cases": "",
                    "proficiency": 0.0,
                    "proficiency_goal": INITIAL_PROFICIENCY_GOAL
                })

        for _, row in scene_grammar.iterrows():
            uid = row["uid"]
            if not ((self.skill["table"] == "grammar") & (self.skill["uid"] == uid)).any():
                new_records.append({
                    "table": "grammar",
                    "uid": uid,
                    "good_cases": "",
                    "bad_cases": "",
                    "proficiency": 0.0,
                    "proficiency_goal": INITIAL_PROFICIENCY_GOAL
                })

        # DEBUG: 测试语法教学能力以及正确完成scene
        """
        for record in new_records:
            #if record['table'] != 'grammar':
            record['proficiency_goal'] = record['proficiency'] = FINAL_PROFICIENCY_GOAL
        """
        
        # 添加到 skill 表中
        if new_records:
            self.skill = pd.concat([self.skill, pd.DataFrame(new_records)], ignore_index=True)


    def load_history(self):
        """
        加载当前scene的主会话context，如果没有则返回空列表
        """
        history_file = self.history_path / f"{self.language}.scene={self.progress['current_scene']}.history.json"
        if history_file.exists():
            with open(history_file, "r", encoding='utf-8') as f:
                return json.loads(f.read())
        else:
            return []
    
    def save_history(self, context):
        """
        保存当前scene的主会话context到指定文件
        """
        if isinstance(self.progress['current_scene'], int):
            history_file = self.history_path / f"{self.language}.scene={self.progress['current_scene']}.history.json"
            with open(history_file, "w", encoding='utf-8') as f:
                # ==== CASCADE PATCH START: fix numpy/pandas json serialization ====
                json.dump(context, f, ensure_ascii=False, indent=2, default=_json_default)
                # ==== CASCADE PATCH END ====
            return True
        else:
            print('[WARNING] invalid current_scene, expected integer')
            return False
    
    def next_scene(self):
        """
        进入下一个场景，更新进度（历史的store和load需要另外处理）
        """
        current_scene = self.progress['current_scene']
        next_scene = current_scene + 1 # 目前只有内置场景，直接按顺序编号
        # 检查是否有下一个场景
        if next_scene in self.scenarios['uid'].values:
            self.progress['current_scene'] = next_scene
            self.progress['scene_completed'] = False
            # 更新skill信息
            self._add_scene_to_skill(next_scene)
            return True
        return False

    
    # 可选方法：用于调试或查看某类数据
    def print_summary(self):
        print(f"Language: {self.language}")
        print(f"Scenarios: {len(self.scenarios)}")
        print(f"Vocab entries: {len(self.vocab)}")
        print(f"Grammar entries: {len(self.grammar)}")
        print(f"World Archive Keys: {list(self.world.keys())}")
        print(f"Skill Archive Keys: {self.skill.info()}")

    # ----- exposed APIs -----
    def get_metadata(self):
        return {
            'scenarios': self.scenarios,
            'vocab': self.vocab,
            'grammar': self.grammar,
            'vocab_emb': self.vocab_emb
        }
    
    def get_archive(self):
        return {
            'world': self.world,
            'skill': self.skill,
            'progress': self.progress
        }
    
    def get_current_scene_status(self):
        # 获取当前scene_uid对应的scene, vocab, grammar子表（由于在当前scene_uid生命周期中不变，故可缓存）
        # 以及skill中的学习状态
        # NOTE: 该方法获得的信息只是存档的副本，对其所作修改无效！

        res = dict()
        if self.status_cache.get('current_scene_uid') == self.progress['current_scene']:
            res['current_vocab'] = self.status_cache.get('current_vocab', None)
            res['current_grammar'] = self.status_cache.get('current_grammar', None)
            res['current_scenario'] = self.status_cache.get('current_scenario', None)
        
        else: # 缓存
            scene_uid = self.progress['current_scene']
            res['current_scenario'] = self.scenarios[self.scenarios['uid'] == scene_uid].copy().iloc[0]
            res['current_vocab'] = self.vocab[self.vocab['scene_uid'] == scene_uid].copy()
            res['current_grammar'] = self.grammar[self.grammar['scene_uid'] == scene_uid].copy()
            self.status_cache['current_scene_uid'] = scene_uid
            self.status_cache['current_vocab'] = res['current_vocab']
            self.status_cache['current_grammar'] = res['current_grammar']
            self.status_cache['current_scenario'] = res['current_scenario']
        
        # 再实时读取当前scene对应的skill，因为其中的proficiency是动态更新的
        res['current_skill'] = self.skill[
            ((self.skill['table'] == 'vocab') & (self.skill['uid'].isin(res['current_vocab']['uid'].tolist()))) | 
            ((self.skill['table'] == 'grammar') & (self.skill['uid'].isin(res['current_grammar']['uid'].tolist())))
        ].copy()

        return res
    
if __name__ == '__main__':
    dm = DataManager('english', debugging=True)
