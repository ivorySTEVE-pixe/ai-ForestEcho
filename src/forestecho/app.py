"""ForestEcho web UI.

Run:
  python -m forestecho.app
  python -m forestecho.app --ckpt models/best.pt --port 7861
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from .features import load_audio
from .model import ASTSpeciesClassifier


NATURE_THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#eafff2",
        c100="#c9fbdc",
        c200="#92f5b8",
        c300="#52e98b",
        c400="#1fd463",
        c500="#08b84b",
        c600="#03953c",
        c700="#067432",
        c800="#0a5a2a",
        c900="#0a3f20",
        c950="#052614",
    ),
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill=(
        "radial-gradient(circle at 12% 8%, #ffe9b8 0%, transparent 38%),"
        "radial-gradient(circle at 88% 14%, #ffc4a8 0%, transparent 42%),"
        "radial-gradient(circle at 78% 92%, #b9f5d6 0%, transparent 48%),"
        "linear-gradient(160deg,#fff8e7 0%,#f0fbe6 45%,#d6f3e3 100%)"
    ),
    block_background_fill="rgba(255,255,255,0.80)",
    block_border_width="1px",
    block_border_color="rgba(8,184,75,0.22)",
    block_shadow="0 8px 24px rgba(8,90,42,0.12)",
    block_radius="14px",
    button_primary_background_fill="linear-gradient(135deg,#08b84b 0%,#1fd463 60%,#52e98b 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg,#03953c 0%,#08b84b 60%,#1fd463 100%)",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=Noto+Serif+JP:wght@500;700&family=Manrope:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700&display=swap');
#fe-hero {
  text-align:center;
  padding: 30px 8px 12px;
  position: relative;
}
#fe-hero h1 {
  margin:0; font-size:3.3rem; font-weight:700; letter-spacing: 0.2px;
  font-family: 'Cormorant Garamond', 'Noto Serif JP', serif;
  background: linear-gradient(115deg,#095d2b 0%,#08b84b 38%,#ffd27a 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  filter: drop-shadow(0 6px 16px rgba(7,80,36,0.14));
}
#fe-hero .tag {
  margin-top: 8px; font-size: 1.15rem; font-style: italic;
  font-family: 'Cormorant Garamond', 'Noto Serif JP', serif;
  color:#3a5736;
}
.fe-card {
  backdrop-filter: blur(12px) saturate(1.08);
  border: 1px solid rgba(255,255,255,0.35);
  box-shadow: 0 14px 30px rgba(9,68,33,0.12), inset 0 1px 0 rgba(255,255,255,0.35);
}
.fe-input-panel {
  background:
    radial-gradient(circle at 9% 6%, rgba(114, 255, 174, 0.14), transparent 40%),
    linear-gradient(165deg, rgba(255,255,255,0.95), rgba(240,255,246,0.86));
  border: 1px solid rgba(8,184,75,0.22);
  box-shadow: 0 18px 34px rgba(7,80,36,0.14), inset 0 1px 0 rgba(255,255,255,0.6);
}
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  padding: 8px 12px 24px !important;
  font-family: 'Manrope', 'Noto Sans JP', sans-serif !important;
  color-scheme: light !important;
}
.gradio-container, body, html {
  background: #0a120c !important;
}
.gradio-container .gr-row {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 12px !important;
}
.gradio-container .gr-column {
  min-width: 320px;
}
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .gr-group {
  background: rgba(16, 26, 19, 0.92) !important;
  border: 1px solid rgba(113, 201, 150, 0.26) !important;
}
.gr-button {
  border-radius: 10px !important;
  font-family: 'Manrope', 'Noto Sans JP', sans-serif !important;
  font-weight: 700 !important;
}
.gr-button-primary {
  box-shadow: 0 10px 20px rgba(8,184,75,0.28) !important;
}
.gr-form, .gr-group {
  border-radius: 12px !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  border-radius: 10px !important;
  border: 1px solid rgba(8,184,75,0.24) !important;
  background: rgba(248,255,251,0.96) !important;
  color: #12291a !important;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
  font-family: 'Manrope', 'Noto Sans JP', sans-serif !important;
}
.gradio-container label,
.gradio-container .gr-form > label,
.gradio-container .gr-input-label,
.gradio-container .gr-block-label {
  font-family: 'Manrope', 'Noto Sans JP', sans-serif !important;
  font-weight: 700 !important;
  color: #d6ebdc !important;
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  display: block !important;
}
.gradio-container .block {
  border-radius: 14px !important;
}
.fe-input-panel .gradio-audio {
  border: 1px solid rgba(8,184,75,0.22);
  border-radius: 14px !important;
  overflow: hidden;
}
.gradio-container .gradio-audio {
  background: rgba(11, 20, 14, 0.9) !important;
}
.gradio-container .gr-dataframe table,
.gradio-container .gr-dataframe th,
.gradio-container .gr-dataframe td {
  background: rgba(14, 23, 17, 0.94) !important;
  color: #e4f2e8 !important;
}
.fe-result-row {
  display:flex; align-items:center; gap:12px; margin:7px 0; padding:12px 13px;
  border-left:4px solid #08b84b; border-radius:10px;
  background: linear-gradient(96deg, rgba(255,255,255,0.96), rgba(235,255,244,0.78));
  box-shadow: 0 4px 14px rgba(7,80,36,0.08);
  transition: transform 0.16s ease, box-shadow 0.16s ease;
}
.fe-result-row:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(7,80,36,0.14);
}
.fe-result-row .name { flex:1; font-weight:700; color:#0f331d; font-size: 1.02rem; }
.fe-result-row .meta { font-size:0.79rem; color:#3d6047; margin-top:2px; }
.fe-result-row .pct { font-weight:700; color:#066a30; width:66px; text-align:right; font-variant-numeric: tabular-nums; }
.fe-result-row .bar { width:34%; height:8px; border-radius:6px; background:rgba(8,184,75,0.14); overflow:hidden; }
.fe-result-row .fill {
  height:100%;
  background: linear-gradient(90deg,#6beea3,#08b84b,#ffd27a);
  box-shadow: 0 0 10px rgba(8,184,75,0.3);
}
.fe-details {
  padding: 14px 14px;
  border-radius: 10px;
  background: linear-gradient(145deg, rgba(238,249,242,0.95), rgba(230,245,236,0.88));
  border: 1px solid rgba(8,184,75,0.16);
}
.fe-details h4 { margin:0 0 6px; color:#0d3a1d; font-size:1.25rem; }
.fe-details .sci { color:#375a43; font-style:italic; margin-bottom:8px; }
.fe-details .line { margin: 4px 0; color:#1f3a29; }
.fe-empty {
  text-align:center; padding: 20px 12px; color:#496141;
  font-style: italic; font-family: 'Cormorant Garamond', 'Noto Serif JP', serif;
}
#fe-footer {
  text-align:center; padding: 18px 0 6px; color:#3f5f47;
  font-style: italic; font-family: 'Cormorant Garamond', 'Noto Serif JP', serif;
}
.meta {
  color: #406046;
}
.fe-input-panel h3 {
  font-weight: 800 !important;
  letter-spacing: 0.1px;
}
.gradio-container h3,
.gradio-container h2,
.gradio-container h1,
.gradio-container p,
.gradio-container span,
.gradio-container div {
  color: inherit;
}
.gradio-container .prose, .gradio-container .markdown, .gradio-container .gr-markdown {
  color: #dceee1 !important;
}
@media (max-width: 980px) {
  #fe-hero h1 { font-size: 2.5rem; }
  #fe-hero .tag { font-size: 1.02rem; }
  .gradio-container { padding: 6px 8px 16px !important; }
  .gradio-container .gr-column { min-width: 100% !important; }
  .fe-result-row { gap: 8px; padding: 10px; }
  .fe-result-row .bar { width: 30%; }
  .fe-result-row .pct { width: 54px; font-size: 0.92rem; }
}
@media (max-width: 640px) {
  #fe-hero h1 { font-size: 2.1rem; }
  #fe-hero .tag { font-size: 0.95rem; }
  .fe-result-row { align-items: flex-start; flex-direction: column; }
  .fe-result-row .bar { width: 100%; }
  .fe-result-row .pct { width: auto; align-self: flex-end; }
  .fe-details h4 { font-size: 1.1rem; }
}
"""


I18N: dict[str, dict[str, str]] = {
    "en": {
        "tagline": "listen to the wild - identify its voices",
        "footer": "every recording is a postcard from the forest.",
        "record": "### Record or Upload",
        "results": "### Prediction Results",
        "confusions": "### Confusion Insights",
        "species_info": "### Species Details",
        "dictionary": "### Species Dictionary",
        "history": "### Recent Detection History",
        "feedback": "### Active Learning Feedback",
        "group_filter": "Taxonomy filter",
        "group_all": "All groups",
        "top_k": "Top predictions",
        "long_audio": "Long audio mode",
        "hop_seconds": "Window hop (seconds)",
        "aggregation": "Aggregation",
        "unknown_threshold": "Unknown threshold",
        "analyze": "Identify species",
        "lang": "日本語",
        "search": "Search species (common/scientific/class code)",
        "select": "Select species",
        "correct_label": "Correct species label",
        "feedback_note": "Optional note",
        "mark_wrong": "Mark prediction as wrong",
        "feedback_saved": "Saved feedback for retraining queue.",
        "feedback_missing": "No prediction/audio context to save yet.",
        "feedback_status_empty": "No feedback saved in this session yet.",
        "empty_audio": "No audio yet - upload or record a clip.",
        "empty_detail": "Select a species to view details.",
        "empty_untrained": "Model is not trained yet - run `python -m forestecho.train` first.",
        "err_process": "Could not process this clip: {err}",
        "status_untrained": "No trained checkpoint found. Train and refresh.",
        "status_loaded": "Loaded {n} species from {path}",
        "group": "Group",
        "scientific": "Scientific Name",
        "description": "Description",
        "habitat": "Habitat",
        "diet": "Diet",
        "wiki": "Learn More",
        "h_time": "Time",
        "h_species": "Species",
        "h_sci": "Scientific",
        "h_conf": "Confidence",
        "unknown_name": "Unknown / Needs review",
        "unknown_desc": "Top confidence is below threshold ({thr:.0f}%). Consider collecting more data for this class.",
    },
    "ja": {
        "tagline": "野生の声に耳をすませ、その種を見いだす",
        "footer": "すべての録音は、森からの絵葉書。",
        "record": "### 録音またはアップロード",
        "results": "### 予測結果",
        "confusions": "### 混同行列インサイト",
        "species_info": "### 種の詳細",
        "dictionary": "### 種辞典",
        "history": "### 検出履歴",
        "feedback": "### 学習フィードバック",
        "group_filter": "分類フィルター",
        "group_all": "すべて",
        "top_k": "上位予測数",
        "long_audio": "長時間音声モード",
        "hop_seconds": "窓の移動幅（秒）",
        "aggregation": "集約方法",
        "unknown_threshold": "未知判定しきい値",
        "analyze": "種を識別する",
        "lang": "English",
        "search": "種を検索（一般名/学名/コード）",
        "select": "種を選択",
        "correct_label": "正しい種ラベル",
        "feedback_note": "メモ（任意）",
        "mark_wrong": "予測ミスとして保存",
        "feedback_saved": "再学習キューに保存しました。",
        "feedback_missing": "保存する予測/音声コンテキストがありません。",
        "feedback_status_empty": "このセッションでの保存はまだありません。",
        "empty_audio": "まだ音声がありません - 録音またはアップロードしてください。",
        "empty_detail": "種を選択すると詳細が表示されます。",
        "empty_untrained": "モデルは未学習です - 先に `python -m forestecho.train` を実行してください。",
        "err_process": "音声処理に失敗しました: {err}",
        "status_untrained": "学習済みチェックポイントがありません。学習後に再読み込みしてください。",
        "status_loaded": "{path} から {n} 種を読み込みました",
        "group": "分類",
        "scientific": "学名",
        "description": "説明",
        "habitat": "生息地",
        "diet": "食性",
        "wiki": "詳細を見る",
        "h_time": "時刻",
        "h_species": "種名",
        "h_sci": "学名",
        "h_conf": "確信度",
        "unknown_name": "不明 / 要確認",
        "unknown_desc": "上位確信度がしきい値（{thr:.0f}%）未満です。この種のデータ追加を検討してください。",
    },
}


SPECIES_DB: dict[str, dict[str, str]] = {
    "crow": {
        "common_en": "Crow",
        "common_ja": "カラス",
        "scientific": "Corvus spp.",
        "group": "Bird",
        "desc_en": "Highly adaptable corvid known for complex social calls.",
        "desc_ja": "社会性が高く、複雑な鳴き声を持つ適応力の高い鳥。",
        "habitat_en": "Urban areas, forests, farmland",
        "habitat_ja": "都市部、森林、農地",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Crow",
    },
    "dog": {
        "common_en": "Dog",
        "common_ja": "イヌ",
        "scientific": "Canis lupus familiaris",
        "group": "Mammal",
        "desc_en": "Domesticated canid with barks, howls, and growls.",
        "desc_ja": "吠え声や遠吠えなど多様な発声を持つ家畜化されたイヌ科。",
        "habitat_en": "Human settlements worldwide",
        "habitat_ja": "人間生活圏全域",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Dog",
    },
    "cat": {
        "common_en": "Cat",
        "common_ja": "ネコ",
        "scientific": "Felis catus",
        "group": "Mammal",
        "desc_en": "Domestic feline with meows, purrs, and trills.",
        "desc_ja": "鳴き声、ゴロゴロ音、トリル音を出す家畜化ネコ科。",
        "habitat_en": "Human settlements worldwide",
        "habitat_ja": "人間生活圏全域",
        "diet_en": "Carnivore",
        "diet_ja": "肉食",
        "wiki": "https://en.wikipedia.org/wiki/Cat",
    },
    "frog": {
        "common_en": "Frog",
        "common_ja": "カエル",
        "scientific": "Anura spp.",
        "group": "Amphibian",
        "desc_en": "Amphibians producing rhythmic calls, often near water.",
        "desc_ja": "主に水辺でリズミカルな鳴き声を出す両生類。",
        "habitat_en": "Wetlands, ponds, riversides",
        "habitat_ja": "湿地、池、河川周辺",
        "diet_en": "Insectivore",
        "diet_ja": "昆虫食",
        "wiki": "https://en.wikipedia.org/wiki/Frog",
    },
    "pig": {
        "common_en": "Pig",
        "common_ja": "ブタ",
        "scientific": "Sus scrofa domesticus",
        "group": "Mammal",
        "desc_en": "Domestic pig with grunts and squeals.",
        "desc_ja": "うなり声や高音の鳴き声を出す家畜化されたブタ。",
        "habitat_en": "Farms and managed landscapes",
        "habitat_ja": "牧場、管理された環境",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Pig",
    },
    "cow": {
        "common_en": "Cow",
        "common_ja": "ウシ",
        "scientific": "Bos taurus",
        "group": "Mammal",
        "desc_en": "Domestic bovine producing low-frequency vocalizations.",
        "desc_ja": "低周波の鳴き声を出す家畜化ウシ。",
        "habitat_en": "Pasture and farms",
        "habitat_ja": "牧草地、農場",
        "diet_en": "Herbivore",
        "diet_ja": "草食",
        "wiki": "https://en.wikipedia.org/wiki/Cattle",
    },
    "sheep": {
        "common_en": "Sheep",
        "common_ja": "ヒツジ",
        "scientific": "Ovis aries",
        "group": "Mammal",
        "desc_en": "Social herbivore with characteristic bleating calls.",
        "desc_ja": "特徴的な鳴き声を持つ群居性の草食哺乳類。",
        "habitat_en": "Pasture and farms",
        "habitat_ja": "牧草地、農場",
        "diet_en": "Herbivore",
        "diet_ja": "草食",
        "wiki": "https://en.wikipedia.org/wiki/Sheep",
    },
    "hen": {
        "common_en": "Hen",
        "common_ja": "メンドリ",
        "scientific": "Gallus gallus domesticus",
        "group": "Bird",
        "desc_en": "Domestic female chicken with clucks and cackles.",
        "desc_ja": "コッコッという声を出す家禽の雌鶏。",
        "habitat_en": "Farms and villages",
        "habitat_ja": "農場、集落周辺",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Chicken",
    },
    "rooster": {
        "common_en": "Rooster",
        "common_ja": "オンドリ",
        "scientific": "Gallus gallus domesticus",
        "group": "Bird",
        "desc_en": "Male chicken known for loud crowing vocalizations.",
        "desc_ja": "大きな時報的な鳴き声で知られる雄鶏。",
        "habitat_en": "Farms and villages",
        "habitat_ja": "農場、集落周辺",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Rooster",
    },
    "insects": {
        "common_en": "Insects",
        "common_ja": "昆虫",
        "scientific": "Insecta",
        "group": "Invertebrate",
        "desc_en": "Broad class of insect stridulation and buzzing sounds.",
        "desc_ja": "羽音や擦音など昆虫由来の広域音クラス。",
        "habitat_en": "Global terrestrial habitats",
        "habitat_ja": "地球上の陸域全般",
        "diet_en": "Varies by species",
        "diet_ja": "種により多様",
        "wiki": "https://en.wikipedia.org/wiki/Insect",
    },
    "mallard": {
        "common_en": "Mallard",
        "common_ja": "マガモ",
        "scientific": "Anas platyrhynchos",
        "group": "Bird",
        "desc_en": "Common duck species with quacking calls.",
        "desc_ja": "クワック音で知られる一般的なカモ種。",
        "habitat_en": "Lakes, ponds, marshes",
        "habitat_ja": "湖、池、湿地",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Mallard",
    },
    "barn_owl": {
        "common_en": "Barn Owl",
        "common_ja": "メンフクロウ",
        "scientific": "Tyto alba",
        "group": "Bird",
        "desc_en": "Nocturnal owl with screech-like calls.",
        "desc_ja": "夜行性で金切り声に近い鳴き声を持つフクロウ。",
        "habitat_en": "Farmland, open woodland, barns",
        "habitat_ja": "農地、疎林、納屋周辺",
        "diet_en": "Carnivore",
        "diet_ja": "肉食",
        "wiki": "https://en.wikipedia.org/wiki/Barn_owl",
    },
    "european_robin": {
        "common_en": "European Robin",
        "common_ja": "ヨーロッパコマドリ",
        "scientific": "Erithacus rubecula",
        "group": "Bird",
        "desc_en": "Small passerine with melodious territorial song.",
        "desc_ja": "さえずりが美しい小型のスズメ目。",
        "habitat_en": "Woodlands, gardens, hedgerows",
        "habitat_ja": "森林、庭園、生垣",
        "diet_en": "Insectivore",
        "diet_ja": "昆虫食",
        "wiki": "https://en.wikipedia.org/wiki/European_robin",
    },
    "common_blackbird": {
        "common_en": "Common Blackbird",
        "common_ja": "クロウタドリ",
        "scientific": "Turdus merula",
        "group": "Bird",
        "desc_en": "Songbird with rich fluting vocalizations.",
        "desc_ja": "豊かで澄んださえずりを持つ鳥。",
        "habitat_en": "Parks, woodland, urban gardens",
        "habitat_ja": "公園、森林、都市庭園",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Common_blackbird",
    },
    "common_cuckoo": {
        "common_en": "Common Cuckoo",
        "common_ja": "カッコウ",
        "scientific": "Cuculus canorus",
        "group": "Bird",
        "desc_en": "Famous two-note call; brood parasite species.",
        "desc_ja": "二音の鳴き声で有名な托卵性の鳥。",
        "habitat_en": "Open woodland, meadows",
        "habitat_ja": "疎林、草地",
        "diet_en": "Insectivore",
        "diet_ja": "昆虫食",
        "wiki": "https://en.wikipedia.org/wiki/Common_cuckoo",
    },
    "great_tit": {
        "common_en": "Great Tit",
        "common_ja": "シジュウカラ",
        "scientific": "Parus major",
        "group": "Bird",
        "desc_en": "Vocal woodland bird with repeated calls and songs.",
        "desc_ja": "反復的な声を持つ森林性の小鳥。",
        "habitat_en": "Woodland, parks, gardens",
        "habitat_ja": "森林、公園、庭園",
        "diet_en": "Omnivore",
        "diet_ja": "雑食",
        "wiki": "https://en.wikipedia.org/wiki/Great_tit",
    },
    "petpet_song": {
        "common_en": "Rock Sparrow (song)",
        "common_ja": "イワスズメ（さえずり）",
        "scientific": "Petronia petronia",
        "group": "Bird",
        "desc_en": "Passerine song class derived from training source code 'petpet_song'.",
        "desc_ja": "学習データコード 'petpet_song' 由来のスズメ目さえずりクラス。",
        "habitat_en": "Rocky open areas, dry scrub, farmland edges",
        "habitat_ja": "岩場の開放地、乾性低木地、農地周辺",
        "diet_en": "Seeds and insects",
        "diet_ja": "種子・昆虫",
        "wiki": "https://en.wikipedia.org/wiki/Rock_sparrow",
    },
    "erirub_call": {
        "common_en": "European Robin (call)",
        "common_ja": "ヨーロッパコマドリ（地鳴き）",
        "scientific": "Erithacus rubecula",
        "group": "Bird",
        "desc_en": "Call vocalization class derived from source label 'erirub_call'.",
        "desc_ja": "元ラベル 'erirub_call' 由来の地鳴きクラス。",
        "habitat_en": "Woodland, gardens",
        "habitat_ja": "森林、庭園",
        "diet_en": "Insectivore",
        "diet_ja": "昆虫食",
        "wiki": "https://en.wikipedia.org/wiki/European_robin",
    },
    "sylmel_song": {
        "common_en": "Sardinian Warbler (song)",
        "common_ja": "サルデーニャムシクイ（さえずり）",
        "scientific": "Curruca melanocephala",
        "group": "Bird",
        "desc_en": "Song vocalization class derived from source label 'sylmel_song'.",
        "desc_ja": "元ラベル 'sylmel_song' 由来のさえずりクラス。",
        "habitat_en": "Mediterranean scrub and thickets",
        "habitat_ja": "地中海性の低木林",
        "diet_en": "Insects and berries",
        "diet_ja": "昆虫・果実",
        "wiki": "https://en.wikipedia.org/wiki/Sardinian_warbler",
    },
    "butbut_call": {
        "common_en": "Butorides-type Heron (call)",
        "common_ja": "ササゴイ類（地鳴き）",
        "scientific": "Butorides spp.",
        "group": "Bird",
        "desc_en": "Call class from source label 'butbut_call'; exact species may vary by dataset source.",
        "desc_ja": "元ラベル 'butbut_call' の地鳴きクラス。データ源により種が異なる場合があります。",
        "habitat_en": "Wetlands, riverbanks, mangrove edges",
        "habitat_ja": "湿地、河川沿い、マングローブ周辺",
        "diet_en": "Fish and aquatic invertebrates",
        "diet_ja": "魚類・水生無脊椎動物",
        "wiki": "https://en.wikipedia.org/wiki/Butorides",
    },
}


def load_species_catalog(path: str | Path) -> dict[str, dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            cleaned[key] = {str(k): str(v) for k, v in value.items()}
    return cleaned


# Prefer external catalog for easy editing; fallback to in-code defaults.
_external_catalog = load_species_catalog("configs/species_catalog.json")
if _external_catalog:
    SPECIES_DB = _external_catalog


def t(lang: str, key: str, **kwargs) -> str:
    text = I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text


def fallback_species(raw: str) -> dict[str, str]:
    label = raw.replace("_", " ").strip()
    title = " ".join(p.capitalize() for p in label.split())
    return {
        "common_en": title,
        "common_ja": title,
        "scientific": "Unknown / needs mapping",
        "group": "Unknown",
        "desc_en": "No curated species profile yet.",
        "desc_ja": "まだ辞書情報が登録されていません。",
        "habitat_en": "Unknown",
        "habitat_ja": "不明",
        "diet_en": "Unknown",
        "diet_ja": "不明",
        "wiki": f"https://www.google.com/search?q={raw}+species",
    }


def species_info(raw: str) -> dict[str, str]:
    return SPECIES_DB.get(raw, fallback_species(raw))


def species_name(raw: str, lang: str) -> str:
    info = species_info(raw)
    return info["common_ja"] if lang == "ja" else info["common_en"]


def species_option(raw: str, lang: str) -> str:
    info = species_info(raw)
    local = info["common_ja"] if lang == "ja" else info["common_en"]
    return f"{local} ({info['scientific']}) :: {raw}"


def all_groups(classes: list[str]) -> list[str]:
    groups = sorted({species_info(raw)["group"] for raw in classes})
    return groups


def parse_raw_from_option(option: str | None) -> str | None:
    if not option or "::" not in option:
        return None
    return option.split("::", 1)[1].strip()


def hero_html(lang: str) -> str:
    return (
        '<div id="fe-hero">'
        "<h1>ForestEcho</h1>"
        f'<div class="tag">{t(lang, "tagline")}</div>'
        "</div>"
    )


def footer_html(lang: str) -> str:
    return f'<div id="fe-footer">{t(lang, "footer")}</div>'


class Predictor:
    def __init__(self, ckpt_path: str | None):
        self.ready = False
        self.ckpt_path = ckpt_path
        self.model: ASTSpeciesClassifier | None = None
        self.classes: list[str] = []
        self.cfg: dict = {}
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if not ckpt_path or not Path(ckpt_path).exists():
            return

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = ckpt["cfg"]
        self.classes = ckpt["classes"]
        self.model = ASTSpeciesClassifier(
            num_classes=len(self.classes),
            pretrained=self.cfg["model"]["pretrained"],
            freeze_backbone=True,
            dropout=self.cfg["model"]["dropout"],
        ).to(self.device)
        self.model.head.load_state_dict(ckpt["head"])
        self.model.eval()
        self.ready = True

    def predict(self, audio_path: str, top_k: int = 5) -> list[tuple[str, float]]:
        assert self.ready and self.model is not None
        sr = self.cfg["data"]["sample_rate"]
        clip = self.cfg["data"]["clip_seconds"]
        waveform = load_audio(audio_path, sr, clip).astype(np.float32)
        model_input = self.model.preprocess([waveform], device=self.device)
        with torch.no_grad():
            probs = self.model(model_input).softmax(-1)[0].cpu()
        k = min(top_k, len(self.classes))
        top = torch.topk(probs, k=k)
        return [(self.classes[idx], float(prob)) for prob, idx in zip(top.values, top.indices)]

    def predict_long(
        self,
        audio_path: str,
        top_k: int = 5,
        hop_seconds: float = 1.5,
        aggregation: str = "mean",
    ) -> list[tuple[str, float]]:
        assert self.ready and self.model is not None
        sr = int(self.cfg["data"]["sample_rate"])
        clip = float(self.cfg["data"]["clip_seconds"])
        y = load_audio(audio_path, sr, clip_seconds=None).astype(np.float32)
        windows = window_waveform(y, sr, clip_seconds=clip, hop_seconds=hop_seconds)
        model_input = self.model.preprocess(windows, device=self.device)
        with torch.no_grad():
            probs = self.model(model_input).softmax(-1).cpu()

        mode = aggregation.lower().strip()
        if mode == "max":
            agg = probs.max(dim=0).values
        elif mode == "vote":
            votes = torch.zeros(probs.shape[1], dtype=torch.float32)
            winners = probs.argmax(dim=1)
            for w in winners:
                votes[int(w)] += 1.0
            agg = votes / max(1, probs.shape[0])
        else:
            agg = probs.mean(dim=0)

        k = min(top_k, len(self.classes))
        top = torch.topk(agg, k=k)
        return [(self.classes[idx], float(prob)) for prob, idx in zip(top.values, top.indices)]


def render_results(items: list[tuple[str, float]], lang: str, group_filter: str = "__all__") -> str:
    if not items:
        return f'<div class="fe-empty">{t(lang, "empty_audio")}</div>'
    grouped: dict[str, list[tuple[str, float]]] = {}
    for raw, prob in items:
        group = species_info(raw)["group"]
        grouped.setdefault(group, []).append((raw, prob))
    rows: list[str] = []
    for group in sorted(grouped.keys()):
        if group_filter != "__all__" and group != group_filter:
            continue
        rows.append(f'<div class="meta" style="font-weight:700;margin-top:8px;">{html.escape(group)}</div>')
        for raw, prob in grouped[group]:
            pct = prob * 100
            info = species_info(raw)
            rows.append(
                '<div class="fe-result-row">'
                f'<div style="flex:1;"><div class="name">{html.escape(species_name(raw, lang))}</div>'
                f'<div class="meta">{html.escape(info["scientific"])} | {html.escape(raw)}</div></div>'
                f'<div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>'
                f'<div class="pct">{pct:.1f}%</div>'
                "</div>"
            )
    if not rows:
        return f'<div class="fe-empty">{t(lang, "empty_audio")}</div>'
    return "".join(rows)


def latest_confusion_insights(classes: list[str], lang: str, limit: int = 8) -> str:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return '<div class="fe-empty">No reports yet. Train once to generate confusion insights.</div>'
    runs = sorted([p for p in reports_dir.iterdir() if p.is_dir()])
    if not runs:
        return '<div class="fe-empty">No reports yet. Train once to generate confusion insights.</div>'
    latest = runs[-1]
    cm_path = latest / "test_confusion_matrix.npy"
    if not cm_path.exists():
        cm_path = latest / "val_confusion_matrix.npy"
    if not cm_path.exists():
        return '<div class="fe-empty">No confusion matrix found in latest report.</div>'
    try:
        cm = np.load(cm_path)
    except Exception:
        return '<div class="fe-empty">Could not load confusion matrix file.</div>'
    n = min(cm.shape[0], len(classes))
    if n == 0:
        return '<div class="fe-empty">No class data in confusion matrix.</div>'

    pairs: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((c, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    if not pairs:
        return '<div class="fe-empty">No off-diagonal confusions found in latest report.</div>'

    rows = []
    for c, i, j in pairs[:limit]:
        true_raw = classes[i]
        pred_raw = classes[j]
        true_name = species_name(true_raw, lang)
        pred_name = species_name(pred_raw, lang)
        rows.append(
            f'<div class="fe-result-row">'
            f'<div style="flex:1;"><div class="name">{html.escape(true_name)} -> {html.escape(pred_name)}</div>'
            f'<div class="meta">{html.escape(true_raw)} -> {html.escape(pred_raw)}</div></div>'
            f'<div class="pct">{c}</div>'
            f"</div>"
        )
    return "".join(rows)


def save_feedback_record(record: dict) -> None:
    out_dir = Path("data/feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corrections.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_unknown_notice(is_unknown: bool, threshold: float, lang: str) -> str:
    if not is_unknown:
        return ""
    return (
        '<div class="fe-empty" style="font-style:normal;">'
        f'<b>{t(lang, "unknown_name")}</b><br>{t(lang, "unknown_desc", thr=threshold*100)}'
        "</div>"
    )


def render_species_details(raw: str | None, lang: str) -> str:
    if not raw:
        return f'<div class="fe-empty">{t(lang, "empty_detail")}</div>'
    info = species_info(raw)
    common = info["common_ja"] if lang == "ja" else info["common_en"]
    desc = info["desc_ja"] if lang == "ja" else info["desc_en"]
    habitat = info["habitat_ja"] if lang == "ja" else info["habitat_en"]
    diet = info["diet_ja"] if lang == "ja" else info["diet_en"]
    wiki = html.escape(info["wiki"])
    return (
        '<div class="fe-details">'
        f"<h4>{html.escape(common)}</h4>"
        f'<div class="sci">{html.escape(info["scientific"])} | code: {html.escape(raw)}</div>'
        f'<div class="line"><b>{t(lang, "group")}:</b> {html.escape(info["group"])}</div>'
        f'<div class="line"><b>{t(lang, "description")}:</b> {html.escape(desc)}</div>'
        f'<div class="line"><b>{t(lang, "habitat")}:</b> {html.escape(habitat)}</div>'
        f'<div class="line"><b>{t(lang, "diet")}:</b> {html.escape(diet)}</div>'
        f'<div class="line"><a href="{wiki}" target="_blank">{t(lang, "wiki")}</a></div>'
        "</div>"
    )


def status_html(predictor: Predictor, lang: str) -> str:
    if predictor.ready:
        msg = t(lang, "status_loaded", n=len(predictor.classes), path=predictor.ckpt_path)
        color = "#067432"
    else:
        msg = t(lang, "status_untrained")
        color = "#a0763a"
    dot = (
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{color};margin-right:8px;"></span>'
    )
    return f'<div style="font-size:0.9rem;color:#2f4c3f;">{dot}{msg}</div>'


def window_waveform(y: np.ndarray, sr: int, clip_seconds: float, hop_seconds: float) -> list[np.ndarray]:
    clip = max(1, int(sr * clip_seconds))
    hop = max(1, int(sr * hop_seconds))
    if len(y) <= clip:
        if len(y) < clip:
            y = np.pad(y, (0, clip - len(y)))
        return [y.astype("float32")]
    windows: list[np.ndarray] = []
    for start in range(0, len(y) - clip + 1, hop):
        windows.append(y[start : start + clip].astype("float32"))
    return windows


def history_table(history: list[dict[str, str | float]], lang: str):
    rows = []
    for row in history[-15:][::-1]:
        rows.append([row["time"], row["species"], row["scientific"], f"{row['conf']:.1f}%"])
    headers = [t(lang, "h_time"), t(lang, "h_species"), t(lang, "h_sci"), t(lang, "h_conf")]
    return rows, headers


def build_app(ckpt_path: str | None) -> gr.Blocks:
    predictor = Predictor(ckpt_path)

    all_classes = predictor.classes if predictor.ready else sorted(SPECIES_DB.keys())
    default_raw = all_classes[0] if all_classes else None

    def options_for(lang: str, classes: list[str]) -> list[str]:
        return [species_option(raw, lang) for raw in classes]

    def search_species(query: str, lang: str):
        q = (query or "").strip().lower()
        pool = all_classes
        if q:
            filtered = []
            for raw in pool:
                info = species_info(raw)
                if (
                    q in raw.lower()
                    or q in info["common_en"].lower()
                    or q in info["common_ja"].lower()
                    or q in info["scientific"].lower()
                ):
                    filtered.append(raw)
            pool = filtered
        opts = options_for(lang, pool)
        if not opts:
            return gr.update(choices=[], value=None), render_species_details(None, lang), None
        return gr.update(choices=opts, value=opts[0]), render_species_details(parse_raw_from_option(opts[0]), lang), parse_raw_from_option(opts[0])

    def select_species(option: str | None, lang: str):
        raw = parse_raw_from_option(option)
        return render_species_details(raw, lang), raw

    def feedback_status(msg: str) -> str:
        return f'<div class="meta" style="margin-top:6px;">{html.escape(msg)}</div>'

    def analyze(audio_path, top_k, group_filter, long_audio, hop_seconds, aggregation, unknown_threshold, lang, hist_state):
        if audio_path is None:
            return (
                render_results([], lang, group_filter),
                "",
                render_species_details(None, lang),
                *history_table(hist_state, lang),
                hist_state,
                None,
                audio_path,
            )
        if not predictor.ready:
            return (
                f'<div class="fe-empty">{t(lang, "empty_untrained")}</div>',
                "",
                render_species_details(None, lang),
                *history_table(hist_state, lang),
                hist_state,
                None,
                audio_path,
            )
        try:
            if long_audio:
                results = predictor.predict_long(
                    audio_path,
                    top_k=int(top_k),
                    hop_seconds=float(hop_seconds),
                    aggregation=str(aggregation),
                )
            else:
                results = predictor.predict(audio_path, top_k=int(top_k))
        except Exception as exc:  # noqa: BLE001
            err = html.escape(str(exc))
            return (
                f'<div class="fe-empty">{t(lang, "err_process", err=err)}</div>',
                "",
                render_species_details(None, lang),
                *history_table(hist_state, lang),
                hist_state,
                None,
                audio_path,
            )

        top_raw, top_prob = results[0]
        unknown = top_prob < float(unknown_threshold)
        top_info = species_info(top_raw)
        pred_name = t(lang, "unknown_name") if unknown else species_name(top_raw, lang)
        hist_state = hist_state + [
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "species": pred_name,
                "scientific": top_info["scientific"],
                "conf": top_prob * 100,
            }
        ]
        return (
            render_results(results, lang, group_filter),
            render_unknown_notice(unknown, float(unknown_threshold), lang),
            render_species_details(None if unknown else top_raw, lang),
            *history_table(hist_state, lang),
            hist_state,
            top_raw,
            audio_path,
        )

    def mark_wrong(
        correct_option,
        note,
        lang,
        last_pred_raw,
        last_audio_path,
        unknown_threshold,
        aggregation,
        long_audio,
        hop_seconds,
    ):
        correct_raw = parse_raw_from_option(correct_option)
        if (not last_pred_raw) or (not last_audio_path) or (not correct_raw):
            return feedback_status(t(lang, "feedback_missing"))
        rec = {
            "timestamp": datetime.now().isoformat(),
            "audio_path": str(last_audio_path),
            "predicted_raw": str(last_pred_raw),
            "predicted_name_en": species_name(str(last_pred_raw), "en"),
            "predicted_name_ja": species_name(str(last_pred_raw), "ja"),
            "correct_raw": str(correct_raw),
            "correct_name_en": species_name(str(correct_raw), "en"),
            "correct_name_ja": species_name(str(correct_raw), "ja"),
            "note": (note or "").strip(),
            "settings": {
                "unknown_threshold": float(unknown_threshold),
                "aggregation": str(aggregation),
                "long_audio": bool(long_audio),
                "hop_seconds": float(hop_seconds),
            },
        }
        save_feedback_record(rec)
        return feedback_status(t(lang, "feedback_saved"))

    def toggle_language(current_lang, hist_state, detail_raw):
        lang = "ja" if current_lang == "en" else "en"
        rows, headers = history_table(hist_state, lang)
        return (
            lang,
            hero_html(lang),
            footer_html(lang),
            gr.update(value=t(lang, "record")),
            gr.update(value=t(lang, "results")),
            gr.update(value=t(lang, "species_info")),
            gr.update(value=t(lang, "dictionary")),
            gr.update(value=t(lang, "history")),
            gr.update(value=t(lang, "confusions")),
            gr.update(value=t(lang, "feedback")),
            gr.update(label=t(lang, "group_filter"), choices=[(t(lang, "group_all"), "__all__")] + [(g, g) for g in all_groups(all_classes)]),
            gr.update(label=t(lang, "top_k")),
            gr.update(label=t(lang, "long_audio")),
            gr.update(label=t(lang, "hop_seconds")),
            gr.update(label=t(lang, "aggregation")),
            gr.update(label=t(lang, "unknown_threshold")),
            gr.update(value=t(lang, "analyze")),
            gr.update(value=t(lang, "lang")),
            gr.update(label=t(lang, "search")),
            gr.update(label=t(lang, "select"), choices=options_for(lang, all_classes)),
            gr.update(label=t(lang, "correct_label"), choices=options_for(lang, all_classes)),
            gr.update(label=t(lang, "feedback_note")),
            gr.update(value=t(lang, "mark_wrong")),
            status_html(predictor, lang),
            latest_confusion_insights(all_classes, lang),
            feedback_status(t(lang, "feedback_status_empty")),
            "",
            render_species_details(detail_raw, lang),
            rows,
            gr.update(headers=headers),
        )

    with gr.Blocks(title="ForestEcho") as app:
        lang_state = gr.State("en")
        history_state = gr.State([])
        detail_raw_state = gr.State(default_raw)
        last_pred_raw_state = gr.State(None)
        last_audio_path_state = gr.State(None)

        with gr.Row():
            lang_btn = gr.Button(t("en", "lang"), size="sm", scale=0, min_width=110)

        hero = gr.HTML(hero_html("en"))

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["fe-card", "fe-input-panel"]):
                section_record = gr.Markdown(t("en", "record"))
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    show_label=False,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#08b84b",
                        waveform_progress_color="#ff7a59",
                        show_recording_waveform=True,
                    ),
                )
                top_k = gr.Slider(1, 10, value=5, step=1, label=t("en", "top_k"))
                long_audio = gr.Checkbox(label=t("en", "long_audio"), value=True)
                hop_seconds = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label=t("en", "hop_seconds"))
                aggregation = gr.Dropdown(
                    label=t("en", "aggregation"),
                    choices=["mean", "max", "vote"],
                    value="mean",
                    interactive=True,
                )
                unknown_threshold = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label=t("en", "unknown_threshold"))
                analyze_btn = gr.Button(t("en", "analyze"), variant="primary", size="lg")
                status = gr.HTML(status_html(predictor, "en"))

            with gr.Column(scale=1, elem_classes="fe-card"):
                section_results = gr.Markdown(t("en", "results"))
                group_filter = gr.Dropdown(
                    label=t("en", "group_filter"),
                    choices=[(t("en", "group_all"), "__all__")] + [(g, g) for g in all_groups(all_classes)],
                    value="__all__",
                    interactive=True,
                )
                results_html = gr.HTML(render_results([], "en", "__all__"))
                unknown_notice = gr.HTML("")
                section_species_info = gr.Markdown(t("en", "species_info"))
                detail_html = gr.HTML(render_species_details(default_raw, "en"))

        with gr.Row():
            with gr.Column(scale=1, elem_classes="fe-card"):
                section_dictionary = gr.Markdown(t("en", "dictionary"))
                search_box = gr.Textbox(label=t("en", "search"), placeholder="crow / Corvus / petpet_song")
                species_select = gr.Dropdown(
                    label=t("en", "select"),
                    choices=options_for("en", all_classes),
                    value=species_option(default_raw, "en") if default_raw else None,
                    interactive=True,
                )
            with gr.Column(scale=1, elem_classes="fe-card"):
                section_confusions = gr.Markdown(t("en", "confusions"))
                confusion_html = gr.HTML(latest_confusion_insights(all_classes, "en"))
            with gr.Column(scale=1, elem_classes="fe-card"):
                section_feedback = gr.Markdown(t("en", "feedback"))
                correct_species_select = gr.Dropdown(
                    label=t("en", "correct_label"),
                    choices=options_for("en", all_classes),
                    value=species_option(default_raw, "en") if default_raw else None,
                    interactive=True,
                )
                feedback_note = gr.Textbox(label=t("en", "feedback_note"), placeholder="e.g. wind + distant call overlap")
                mark_wrong_btn = gr.Button(t("en", "mark_wrong"), variant="secondary")
                feedback_status_html = gr.HTML(feedback_status(t("en", "feedback_status_empty")))
            with gr.Column(scale=1, elem_classes="fe-card"):
                section_history = gr.Markdown(t("en", "history"))
                history_rows, history_headers = history_table([], "en")
                history_df = gr.Dataframe(
                    headers=history_headers,
                    value=history_rows,
                    row_count=(8, "dynamic"),
                    col_count=(4, "fixed"),
                    interactive=False,
                    wrap=True,
                )

        footer = gr.HTML(footer_html("en"))

        analyze_btn.click(
            analyze,
            inputs=[audio, top_k, group_filter, long_audio, hop_seconds, aggregation, unknown_threshold, lang_state, history_state],
            outputs=[results_html, unknown_notice, detail_html, history_df, history_df, history_state, last_pred_raw_state, last_audio_path_state],
        )
        audio.change(
            analyze,
            inputs=[audio, top_k, group_filter, long_audio, hop_seconds, aggregation, unknown_threshold, lang_state, history_state],
            outputs=[results_html, unknown_notice, detail_html, history_df, history_df, history_state, last_pred_raw_state, last_audio_path_state],
        )
        group_filter.change(
            analyze,
            inputs=[audio, top_k, group_filter, long_audio, hop_seconds, aggregation, unknown_threshold, lang_state, history_state],
            outputs=[results_html, unknown_notice, detail_html, history_df, history_df, history_state, last_pred_raw_state, last_audio_path_state],
        )
        search_box.change(
            search_species,
            inputs=[search_box, lang_state],
            outputs=[species_select, detail_html, detail_raw_state],
        )
        species_select.change(
            select_species,
            inputs=[species_select, lang_state],
            outputs=[detail_html, detail_raw_state],
        )
        mark_wrong_btn.click(
            mark_wrong,
            inputs=[
                correct_species_select,
                feedback_note,
                lang_state,
                last_pred_raw_state,
                last_audio_path_state,
                unknown_threshold,
                aggregation,
                long_audio,
                hop_seconds,
            ],
            outputs=[feedback_status_html],
        )
        lang_btn.click(
            toggle_language,
            inputs=[lang_state, history_state, detail_raw_state],
            outputs=[
                lang_state,
                hero,
                footer,
                section_record,
                section_results,
                section_species_info,
                section_dictionary,
                section_history,
                section_confusions,
                section_feedback,
                group_filter,
                top_k,
                long_audio,
                hop_seconds,
                aggregation,
                unknown_threshold,
                analyze_btn,
                lang_btn,
                search_box,
                species_select,
                correct_species_select,
                feedback_note,
                mark_wrong_btn,
                status,
                confusion_html,
                feedback_status_html,
                unknown_notice,
                detail_html,
                history_df,
                history_df,
            ],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.ckpt)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=NATURE_THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
