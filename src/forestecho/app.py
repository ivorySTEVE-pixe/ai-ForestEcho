"""ForestEcho web UI.

Run:
  python -m forestecho.app
  python -m forestecho.app --ckpt models/best.pt --port 7861
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap');
.gradio-container{max-width:100% !important;padding:0 !important;font-family:'DM Sans',sans-serif !important;background:#f2f5f3 !important;}
.gradio-container .block,.gradio-container .gr-box,.gradio-container .gr-panel,.gradio-container .gr-form,.gradio-container .gr-group{background:transparent !important;border:none !important;box-shadow:none !important;border-radius:0 !important;}
.gradio-container .gr-row{gap:20px !important;}
.gradio-container h3,.gradio-container .markdown h3{margin:0;}

#fe-hero-shell{
  background:
    linear-gradient(90deg,rgba(7,22,15,.88) 0%,rgba(7,28,20,.7) 52%,rgba(9,32,24,.35) 100%),
    url('https://images.unsplash.com/photo-1448375240586-882707db888b?q=80&w=2200&auto=format&fit=crop');
  background-size:cover;background-position:center;
  min-height:620px;padding:10px 0 28px;
}
#fe-header{
  height:68px;border-radius:0;background:rgba(5,19,13,.84);
  display:grid;grid-template-columns:320px 1fr 190px;align-items:center;
  gap:18px;padding:0 14px 0 0;margin:0 auto 28px;max-width:1720px;
  box-shadow:0 10px 24px rgba(0,0,0,.22);
}
.fe-brand{display:flex;align-items:center;gap:12px;color:#fff;padding-left:8px;}
.fe-brand .dot{
  width:48px;height:48px;border-radius:14px;
  background:linear-gradient(145deg,#237a47,#1a5f39);
  color:#cff7de;display:inline-flex;align-items:center;justify-content:center;font-size:20px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.2), 0 8px 14px rgba(0,0,0,.24);
}
.fe-brand span{font-size:40px;font-weight:700;line-height:1;}
.fe-brand small{display:block;font-size:12px;color:#82bb97;letter-spacing:.02em;margin-top:2px;}
.fe-nav{display:flex;gap:34px;justify-content:center;color:#9ec0ab;font-size:18px;font-weight:600;}
.fe-nav span{padding:4px 0;border-bottom:2px solid transparent;transition:color .16s ease,border-color .16s ease;}
.fe-nav span:hover{color:#cde8d8;}
.fe-nav span.active{color:#35c56f;border-bottom-color:#35c56f;}
.fe-badge{
  justify-self:end;background:rgba(14,49,31,.7);border:1px solid rgba(66,143,95,.45);color:#34b565;
  padding:8px 16px;border-radius:24px;font-size:16px;font-weight:700;display:flex;align-items:center;gap:10px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.12), 0 6px 12px rgba(0,0,0,.16);
}
.live-dot{width:10px;height:10px;border-radius:50%;background:#3ed27a;display:inline-block;}

#fe-hero-grid{display:grid !important;grid-template-columns:1.35fr 1fr;gap:28px;align-items:start;max-width:1720px;margin:0 auto;}
#fe-hero-copy{padding-top:66px;max-width:760px;}
#fe-hero-copy h1{font-family:'DM Serif Display',serif;font-size:80px;line-height:1.02;font-weight:400;color:#ecf7ef;margin:0 0 22px;}
#fe-hero-copy h1 span{color:#1f8f4f;font-family:'DM Sans',sans-serif;font-weight:700;}
#fe-hero-copy p{font-size:19px;line-height:1.5;color:#c2d8cd;margin:0 0 32px;max-width:700px;}
.hero-features{display:flex;gap:28px;flex-wrap:wrap;}
.hf-item{max-width:210px;color:#c7ddcf;}
.hf-item b{display:block;color:#f5fff9;font-size:34px;font-weight:700;margin-bottom:2px;}
.hf-item span{font-size:14px;line-height:1.35;color:#97baaa;}

#fe-upload-card{
  margin-top:44px;background:#d9e6de;border:1px solid #9ddbb7;border-radius:26px;padding:22px;max-width:560px;justify-self:end;
  box-shadow:0 18px 28px rgba(4,17,11,.2);
}
#fe-upload-card .markdown h3{color:#173024 !important;font-size:40px !important;font-weight:700 !important;}
.fe-upload-sub{font-size:14px;color:#496f5a;margin:6px 0 16px;}
#fe-upload-card .gradio-audio{
  background:#d5e4dc !important;border:2px dashed #93e2ad !important;border-radius:20px !important;
  min-height:220px;padding:16px;
}
#fe-upload-card .gradio-audio *{color:#1f3528 !important;}
#fe-upload-card .gr-button-primary{
  margin-top:14px;background:#1b6f3f !important;border:none !important;color:#fff !important;
  border-radius:14px !important;font-size:18px !important;font-weight:700 !important;padding:12px 16px !important;
  box-shadow:0 8px 18px rgba(27,111,63,.35);
  transition:transform .14s ease, box-shadow .14s ease, background .14s ease;
}
#fe-upload-card .gr-button-primary:hover{
  background:#155a32 !important;
  transform:translateY(-1px);
  box-shadow:0 12px 22px rgba(21,90,50,.42);
}
.fe-record-row{
  margin-top:12px;padding:12px 16px;border-radius:18px;border:2px solid #97e1b0;background:#d7e5dc;
  color:#1d3a2b;font-size:17px;font-weight:700;text-align:center;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.55);
}
.fe-local-note{margin-top:10px;text-align:center;color:#4f7a64;font-size:14px;}
.fe-analysis-msg{margin-top:8px;text-align:center;color:#1a5a35;font-size:14px;min-height:20px;font-weight:600;}

#fe-content{padding:26px 0 18px;background:#f2f5f3;max-width:1720px;margin:0 auto;}
#fe-mid-wrap{
  border:2px solid #95deaf;
  border-radius:34px;
  padding:28px;
  background:#d9e8df;
}
#fe-mid-wrap .gr-row{align-items:stretch !important;}
.section-card{
  background:linear-gradient(155deg,#79df97,#9be9b1);
  border:2px solid #67d88c;border-radius:24px;padding:26px;min-height:420px;height:100%;
  box-shadow:0 8px 18px rgba(43,144,83,.18);
}
.section-card h3{font-size:42px;color:#123125;margin:0 0 18px;padding-left:14px;border-left:5px solid #167f44;}
.steps{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:8px;align-items:start;}
.step{text-align:center;}
.step .ico{width:74px;height:74px;border-radius:18px;border:1px solid #bdebcf;background:#d6f0de;display:flex;align-items:center;justify-content:center;margin:0 auto 12px;font-weight:700;color:#1f7f47;font-size:28px;}
.step h4{font-size:17px;margin:0 0 5px;color:#123126;}
.step p{font-size:14px;color:#38604a;margin:0;line-height:1.45;}
.arrow{font-size:35px;color:#286745;margin-top:14px;}

#fe-recent-card{
  background:#d4e5db;border:2px solid #66d58a;border-radius:24px;padding:0 24px 16px;overflow:hidden;min-height:420px;height:100%;
  box-shadow:0 8px 18px rgba(44,126,79,.16);
}
.rp-cover{height:124px;margin:0 -24px 12px;background:url('https://images.unsplash.com/photo-1444464666168-49d633b86797?q=80&w=1200&auto=format&fit=crop') center/cover no-repeat;opacity:.8;}
.rp-head{display:flex;justify-content:space-between;align-items:center;margin:0 0 8px;}
#fe-recent-card h3{font-size:42px;color:#132f24;margin:0;padding-left:14px;border-left:5px solid #1f7b47;}
.rp-link{font-size:14px;color:#2e7a4f;font-weight:700;}
.rp-row{display:grid;grid-template-columns:56px 1fr 220px;gap:14px;align-items:center;padding:12px 0;border-top:1px solid #9fdfb8;}
.rp-row:first-of-type{border-top:none;}
.rp-avatar{width:56px;height:56px;border-radius:12px;background:linear-gradient(135deg,#2ca4ff,#1e86de);display:flex;align-items:center;justify-content:center;color:#f5fdff;font-weight:700;font-size:20px;}
.rp-avatar img{width:100%;height:100%;object-fit:cover;border-radius:12px;}
.rp-name{font-size:18px;font-weight:700;color:#1a3025;}
.rp-sci{font-size:15px;color:#4f6f5f;font-style:italic;}
.rp-right{text-align:right;}
.rp-pct{font-size:38px;font-weight:700;color:#1f6b41;}
.rp-time{font-size:14px;color:#6b8477;}
.rp-bar{height:6px;background:#b7cfc0;border-radius:8px;margin-top:6px;overflow:hidden;}
.rp-fill{height:100%;background:linear-gradient(90deg,#67be84,#2a7f4a);}

#fe-stats{
  margin:18px 0 14px;display:grid;grid-template-columns:repeat(4,1fr);gap:16px;
}
.stat{
  background:#d5e3db;border:2px solid #93deae;border-radius:26px;padding:22px 24px;text-align:left;
  box-shadow:0 6px 14px rgba(62,136,90,.14);
}
.stat b{display:block;font-size:50px;color:#132e22;line-height:1;}
.stat span{font-size:17px;color:#425c4f;}

#fe-footer{
  background:linear-gradient(90deg,#051b12,#0a2a1b);color:#d2e8dc;
  display:grid;grid-template-columns:1.3fr 1fr 1fr 1fr;gap:18px;padding:30px 64px;
}
#fe-footer h4{font-size:38px;margin:0 0 10px;color:#fff;}
#fe-footer h5{font-size:22px;margin:0 0 10px;color:#9ec3b0;letter-spacing:.05em;text-transform:uppercase;}
#fe-footer p{margin:0;font-size:14px;line-height:1.62;color:#9ab7a8;}

@media (max-width:1200px){
  #fe-hero-shell{padding-left:18px;padding-right:18px;}
  #fe-content{padding-left:18px;padding-right:18px;}
  #fe-footer{padding-left:18px;padding-right:18px;}
  #fe-mid-wrap{padding:14px;border-radius:20px;}
  #fe-hero-grid{grid-template-columns:1fr !important;}
  #fe-hero-copy{padding-top:24px;}
  #fe-upload-card{max-width:none;justify-self:stretch;}
  #fe-header{grid-template-columns:1fr;gap:10px;height:auto;padding:10px;}
  .fe-nav{justify-content:flex-start;flex-wrap:wrap;gap:16px;}
  .fe-badge{justify-self:start;}
  .steps{grid-template-columns:1fr 1fr;}
  .arrow{display:none;}
  #fe-stats{grid-template-columns:1fr 1fr;}
  #fe-footer{grid-template-columns:1fr 1fr;}
}
@media (max-width:720px){
  #fe-hero-copy h1{font-size:52px;}
  #fe-content{padding-top:16px;}
  #fe-stats{grid-template-columns:1fr;}
  #fe-footer{grid-template-columns:1fr;}
}
"""

IFRAME_CSS = """
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
body, html { margin: 0 !important; padding: 0 !important; background: #0a160f !important; }
#design-frame { width: 100vw; height: 100vh; border: 0; display: block; }
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


def topbar_html() -> str:
    return (
        '<div id="fe-header">'
        '<div class="fe-brand"><span class="dot">◉</span><span>ForestEcho</span><small>Local • Private • Accurate</small></div>'
        '<div class="fe-nav"><span class="active">Home</span><span>About</span><span>Model</span><span>Species</span><span>How It Works</span><span>Contact</span></div>'
        '<div class="fe-badge"><span class="live-dot"></span>Local Model</div>'
        "</div>"
    )


def hero_html(lang: str) -> str:
    return (
        '<div id="fe-hero-copy">'
        '<div style="color:#2fb464;font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:14px;margin-bottom:14px;">AI Sound Recognition</div>'
        '<h1>Identify Animals<br>by Their <span>Sound</span></h1>'
        "<p>Upload an audio clip and our locally trained AI model will predict the type and species of animal.</p>"
        '<div class="hero-features">'
        '<div class="hf-item"><b>100% Local</b><span>Your data stays on device</span></div>'
        '<div class="hf-item"><b>Privacy First</b><span>No cloud uploads</span></div>'
        '<div class="hf-item"><b>Wildlife Focus</b><span>50+ species</span></div>'
        "</div>"
        "</div>"
    )


def how_it_works_html() -> str:
    return (
        '<div class="section-card"><h3>How It Works</h3>'
        '<div class="steps">'
        '<div class="step"><div class="ico">1</div><h4>Upload Audio</h4><p>Upload a recording of an animal sound.</p></div>'
        '<div class="arrow">→</div>'
        '<div class="step"><div class="ico">2</div><h4>Process Audio</h4><p>The audio is processed and analyzed locally.</p></div>'
        '<div class="arrow">→</div>'
        '<div class="step"><div class="ico">3</div><h4>AI Prediction</h4><p>The model predicts animal type and species.</p></div>'
        '<div class="arrow">→</div>'
        '<div class="step"><div class="ico">4</div><h4>Get Results</h4><p>See confidence scores and top matches.</p></div>'
        "</div></div>"
    )


def stats_html() -> str:
    return (
        '<div id="fe-stats">'
        '<div class="stat"><b>50+</b><span>Animal Species</span></div>'
        '<div class="stat"><b>10,000+</b><span>Audio Samples</span></div>'
        '<div class="stat"><b>98%</b><span>Model Accuracy</span></div>'
        '<div class="stat"><b>100%</b><span>Local & Private</span></div>'
        "</div>"
    )


def footer_html() -> str:
    return (
        '<div id="fe-footer">'
        '<div class="f-col"><h4>ForestEcho</h4><p>Local AI model for animal sound recognition.<br>Built for privacy. Made for wildlife.</p></div>'
        '<div class="f-col"><h5>Quick Links</h5><p>Home<br>About<br>Model<br>Species<br>How It Works<br>Contact</p></div>'
        '<div class="f-col"><h5>Model Info</h5><p>Model Type: CNN + Transformer<br>Trained On: Local Wildlife Dataset<br>Running On: Your Device (Local)</p></div>'
        '<div class="f-col"><p>Made with ❤️ for wildlife<br><br>© 2026 ForestEcho. All rights reserved.</p></div>'
        "</div>"
    )


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


HISTORY_PATH = Path("data/history/predictions.jsonl")


def save_prediction_history(
    *,
    raw: str,
    scientific: str,
    confidence: float,
    source_file: str,
) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "raw": raw,
        "scientific": scientific,
        "confidence": round(float(confidence), 2),
        "source_file": source_file,
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prediction_history(limit: int = 10, lang: str = "en") -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = rows[-max(1, limit) :][::-1]
    out: list[dict[str, Any]] = []
    for row in rows:
        raw = str(row.get("raw", ""))
        info = species_info(raw) if raw else fallback_species("unknown")
        out.append(
            {
                "ts": row.get("ts"),
                "raw": raw,
                "name": info["common_ja"] if lang == "ja" else info["common_en"],
                "scientific": row.get("scientific", info["scientific"]),
                "confidence": float(row.get("confidence", 0.0)),
                "source_file": row.get("source_file", ""),
            }
        )
    return out


def find_frontend_files() -> tuple[Path | None, Path | None]:
    candidates = [
        Path("web/ForestEcho.html"),
        Path("/Users/aankansarkar/Downloads/ForestEcho.html"),
    ]
    html_path = next((p for p in candidates if p.exists()), None)
    if html_path is None:
        return None, None
    jsx_path = html_path.parent / "tweaks-panel.jsx"
    return html_path, jsx_path if jsx_path.exists() else None


def build_server(ckpt_path: str | None) -> gr.Server:
    predictor = Predictor(ckpt_path)
    html_path, jsx_path = find_frontend_files()
    app = gr.Server(title="ForestEcho")

    @app.get("/")
    async def index():
        if html_path is None:
            return PlainTextResponse(
                "ForestEcho.html not found. Place it in ./web or ~/Downloads.",
                status_code=404,
            )
        return FileResponse(html_path)

    @app.get("/tweaks-panel.jsx")
    async def tweaks_panel():
        if jsx_path is None:
            return PlainTextResponse("// tweaks-panel.jsx not found", status_code=404)
        return FileResponse(jsx_path)

    @app.get("/api/status")
    async def status():
        return JSONResponse(
            {
                "ready": predictor.ready,
                "classes": len(predictor.classes),
                "checkpoint": predictor.ckpt_path,
            }
        )

    @app.get("/api/history")
    async def history_api(limit: int = 10, lang: str = "en"):
        n = max(1, min(int(limit), 100))
        return JSONResponse({"ok": True, "history": load_prediction_history(limit=n, lang=lang)})

    @app.post("/api/predict")
    async def predict_api(
        audio: UploadFile = File(...),
        top_k: int = Form(3),
        lang: str = Form("en"),
    ):
        if not predictor.ready:
            return JSONResponse({"error": "Model is not loaded."}, status_code=503)

        suffix = Path(audio.filename or "input.wav").suffix or ".wav"
        temp_path: Path | None = None
        converted_path: Path | None = None
        try:
            data = await audio.read()
            if not data:
                return JSONResponse({"error": "Uploaded audio is empty. Please record or upload again."}, status_code=400)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                temp_path = Path(tmp.name)

            k = max(1, min(int(top_k), 10))
            try:
                items = predictor.predict(str(temp_path), top_k=k)
            except Exception:
                # Some mobile-recorded formats (notably iOS m4a variants) decode more reliably
                # after explicit ffmpeg normalization to wav/mono.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    converted_path = Path(tmp_wav.name)
                try:
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(temp_path),
                        "-ac",
                        "1",
                        "-ar",
                        str(int(predictor.cfg["data"]["sample_rate"])),
                        str(converted_path),
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if proc.returncode != 0:
                        err_tail = (proc.stderr or "").strip().splitlines()[-1:] or ["ffmpeg decode failed"]
                        return JSONResponse(
                            {"error": f"Could not decode this recording format. {err_tail[0]}"},
                            status_code=400,
                        )
                    items = predictor.predict(str(converted_path), top_k=k)
                except FileNotFoundError:
                    return JSONResponse(
                        {"error": "ffmpeg is not installed on server. Install ffmpeg for mobile audio compatibility."},
                        status_code=500,
                    )
            payload: list[dict[str, Any]] = []
            for raw, prob in items:
                info = species_info(raw)
                local_name = info["common_ja"] if lang == "ja" else info["common_en"]
                desc = info["desc_ja"] if lang == "ja" else info["desc_en"]
                payload.append(
                    {
                        "raw": raw,
                        "name": local_name,
                        "scientific": info["scientific"],
                        "group": info["group"],
                        "description": desc,
                        "confidence": round(float(prob) * 100.0, 2),
                        "wiki": info["wiki"],
                    }
                )

            if payload:
                top = payload[0]
                save_prediction_history(
                    raw=str(top["raw"]),
                    scientific=str(top["scientific"]),
                    confidence=float(top["confidence"]),
                    source_file=audio.filename or "recorded_audio",
                )

            return JSONResponse(
                {
                    "ok": True,
                    "ready": True,
                    "predictions": payload,
                    "top": payload[0] if payload else None,
                }
            )
        except Exception as exc:
            return JSONResponse({"error": f"Prediction failed: {exc}"}, status_code=500)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if converted_path is not None:
                converted_path.unlink(missing_ok=True)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_server(args.ckpt)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
