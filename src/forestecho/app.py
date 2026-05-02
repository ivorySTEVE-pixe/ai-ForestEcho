"""ForestEcho web UI — a polished Gradio interface for the species classifier.

Run:   python -m forestecho.app
       python -m forestecho.app --ckpt models/best.pt --share
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml

from .features import load_audio
from .model import ASTSpeciesClassifier


# ---------- Theme & CSS --------------------------------------------------------

NATURE_THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#eafff2", c100="#c9fbdc", c200="#92f5b8", c300="#52e98b",
        c400="#1fd463", c500="#08b84b", c600="#03953c", c700="#067432",
        c800="#0a5a2a", c900="#0a3f20", c950="#052614",
    ),
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill=(
        "radial-gradient(circle at 12% 8%, #ffe9b8 0%, transparent 38%),"
        "radial-gradient(circle at 88% 14%, #ffc4a8 0%, transparent 42%),"
        "radial-gradient(circle at 78% 92%, #b9f5d6 0%, transparent 48%),"
        "linear-gradient(160deg,#fff8e7 0%,#f0fbe6 45%,#d6f3e3 100%)"
    ),
    body_background_fill_dark=(
        "radial-gradient(circle at 15% 10%, #1f4d2c 0%, transparent 45%),"
        "radial-gradient(circle at 85% 90%, #4a2a1a 0%, transparent 50%),"
        "linear-gradient(160deg,#081a0f 0%,#102218 50%,#0a1d14 100%)"
    ),
    block_background_fill="rgba(255,255,255,0.72)",
    block_background_fill_dark="rgba(18,32,22,0.78)",
    block_border_width="1px",
    block_border_color="rgba(8,184,75,0.22)",
    block_shadow="0 10px 32px rgba(8,90,42,0.14), 0 2px 6px rgba(255,148,84,0.08)",
    block_radius="18px",
    button_primary_background_fill="linear-gradient(135deg,#08b84b 0%,#1fd463 50%,#52e98b 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg,#03953c 0%,#08b84b 50%,#1fd463 100%)",
    button_primary_text_color="#ffffff",
    button_primary_shadow="0 6px 18px rgba(8,184,75,0.45)",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Noto+Serif+JP:wght@500;600;700&family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;600;700&display=swap');

@keyframes fe-shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fe-float {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-4px); }
}
@keyframes fe-pulse-glow {
    0%,100% { box-shadow: 0 0 0 0 rgba(8,184,75,0.45); }
    50%     { box-shadow: 0 0 0 12px rgba(8,184,75,0); }
}
@keyframes fe-fill-grow {
    from { width: 0; }
}

#fe-hero { text-align: center; padding: 36px 16px 12px; position: relative; }
#fe-hero h1 {
    font-family: 'Cormorant Garamond', 'Noto Serif JP', serif !important;
    font-size: 3.6rem; font-weight: 700; letter-spacing: 0.5px; margin: 0;
    background: linear-gradient(110deg,#067432 0%,#08b84b 25%,#52e98b 45%,#ffb454 70%,#ff7a59 100%);
    background-size: 250% 250%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: fe-shimmer 9s ease-in-out infinite;
    filter: drop-shadow(0 4px 18px rgba(8,184,75,0.18));
}
#fe-hero h1 > span.fe-leaf {
    -webkit-text-fill-color: initial; display: inline-block;
    animation: fe-float 4s ease-in-out infinite;
}
#fe-hero .fe-tag {
    font-family: 'Cormorant Garamond', 'Noto Serif JP', serif;
    font-style: italic; font-size: 1.25rem;
    background: linear-gradient(90deg,#067432,#ff7a59);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-top: 6px; letter-spacing: 0.4px;
}
#fe-hero .fe-rule {
    width: 110px; height: 3px; margin: 18px auto 0; border-radius: 2px;
    background: linear-gradient(90deg, transparent, #08b84b 30%, #ffb454 70%, transparent);
}

.fe-card { backdrop-filter: blur(10px) saturate(1.15); }

.fe-result-row {
    display: flex; align-items: center; gap: 14px;
    padding: 12px 14px; margin: 8px 0;
    background: linear-gradient(95deg, rgba(255,255,255,0.85), rgba(234,255,242,0.7));
    border-left: 4px solid transparent;
    border-image: linear-gradient(180deg,#08b84b,#52e98b) 1;
    border-radius: 10px;
    font-family: 'Inter', 'Noto Sans JP', sans-serif;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.fe-result-row:hover {
    transform: translateX(3px);
    box-shadow: 0 6px 18px rgba(8,184,75,0.18);
}
.fe-result-row .fe-name { flex: 1; font-weight: 600; color: #0a3f20; font-size: 1rem; }
.fe-result-row .fe-pct  {
    font-variant-numeric: tabular-nums; font-weight: 700;
    background: linear-gradient(90deg,#067432,#08b84b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.fe-result-row .fe-bar  {
    flex-basis: 42%; height: 8px; border-radius: 4px;
    background: rgba(8,184,75,0.10); overflow: hidden;
}
.fe-result-row .fe-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg,#52e98b 0%,#08b84b 50%,#ffb454 100%);
    background-size: 200% 100%;
    animation: fe-shimmer 4s linear infinite, fe-fill-grow 0.7s ease-out;
    box-shadow: 0 0 8px rgba(8,184,75,0.5);
}
.fe-top .fe-result-row {
    background: linear-gradient(95deg, rgba(255,233,184,0.85), rgba(201,251,220,0.85));
    border-image: linear-gradient(180deg,#ffb454,#08b84b) 1;
    border-left-width: 6px;
    box-shadow: 0 6px 22px rgba(255,180,84,0.22);
}
.fe-top .fe-name { font-size: 1.25rem; font-weight: 700; }
.fe-top .fe-pct  { font-size: 1.05rem; }

#fe-footer {
    text-align: center; padding: 22px 0 10px; font-size: 1rem;
    font-family: 'Cormorant Garamond', 'Noto Serif JP', serif; font-style: italic;
    background: linear-gradient(90deg,#067432,#ff7a59);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.fe-empty {
    padding: 32px 20px; text-align: center;
    font-family: 'Cormorant Garamond', 'Noto Serif JP', serif;
    font-style: italic; font-size: 1.1rem;
    background: linear-gradient(90deg,#067432,#ffb454);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.fe-lang-row { display: flex; justify-content: flex-end; padding: 8px 14px 0; }
.fe-lang-row button {
    background: linear-gradient(135deg,#ffb454,#ff7a59) !important;
    color: #ffffff !important; border: none !important;
    box-shadow: 0 4px 12px rgba(255,122,89,0.35) !important;
    font-weight: 600 !important;
}

button.primary, .gr-button-primary { animation: fe-pulse-glow 2.6s ease-in-out infinite; }

h3 {
    background: linear-gradient(90deg,#067432,#08b84b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    font-weight: 700 !important;
}
"""


# ---------- i18n ---------------------------------------------------------------

I18N: dict[str, dict[str, str]] = {
    "en": {
        "tagline": "listen to the wild — identify its voices",
        "footer": "every recording is a postcard from the forest.",
        "section_record": "### 🎙  Record or upload",
        "section_results": "### 🌾  What we hear",
        "slider_top_k": "Top predictions",
        "btn_analyze": "Identify species",
        "btn_lang": "日本語",
        "empty_no_audio": "no audio yet — share a recording above",
        "empty_untrained": "model is not trained yet — run <code>python -m forestecho.train</code> first.",
        "err_process": "could not process this clip: {err}",
        "status_untrained": (
            "No trained checkpoint found yet. Train the model with "
            "<code>python -m forestecho.train</code> and refresh — until then the "
            "interface is in preview mode."
        ),
        "status_loaded": "Loaded {n} species from {path}",
    },
    "ja": {
        "tagline": "野生の声に耳をすませ、その種を見いだす",
        "footer": "すべての録音は、森からの絵葉書。",
        "section_record": "### 🎙  録音またはアップロード",
        "section_results": "### 🌾  解析結果",
        "slider_top_k": "上位予測の数",
        "btn_analyze": "種を識別する",
        "btn_lang": "English",
        "empty_no_audio": "まだ音声がありません — 上から録音を共有してください",
        "empty_untrained": "モデルはまだ学習されていません — まず <code>python -m forestecho.train</code> を実行してください。",
        "err_process": "この音声を処理できませんでした: {err}",
        "status_untrained": (
            "学習済みチェックポイントが見つかりません。"
            "<code>python -m forestecho.train</code> でモデルを学習し、再読み込みしてください — "
            "それまではプレビューモードです。"
        ),
        "status_loaded": "{path} から {n} 種を読み込みました",
    },
}


def t(lang: str, key: str, **kwargs) -> str:
    s = I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))
    return s.format(**kwargs) if kwargs else s


def hero_html(lang: str) -> str:
    return (
        '<div id="fe-hero">'
        '<h1><span class="fe-leaf">🌿</span> ForestEcho</h1>'
        f'<div class="fe-tag">{t(lang, "tagline")}</div>'
        '<div class="fe-rule"></div>'
        '</div>'
    )


def footer_html(lang: str) -> str:
    return f'<div id="fe-footer">{t(lang, "footer")}</div>'


# ---------- Model loading ------------------------------------------------------

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
        y = load_audio(audio_path, sr, clip).astype(np.float32)
        x = self.model.preprocess([y], device=self.device)
        with torch.no_grad():
            probs = self.model(x).softmax(-1)[0].cpu()
        k = min(top_k, len(self.classes))
        top = torch.topk(probs, k=k)
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices)]


# ---------- Rendering ----------------------------------------------------------

def render_results(items: list[tuple[str, float]], lang: str) -> str:
    if not items:
        return f'<div class="fe-empty">{t(lang, "empty_no_audio")}</div>'
    rows = []
    for idx, (name, prob) in enumerate(items):
        pct = prob * 100
        wrap_open = '<div class="fe-top">' if idx == 0 else ""
        wrap_close = "</div>" if idx == 0 else ""
        rows.append(
            f'{wrap_open}<div class="fe-result-row">'
            f'<span class="fe-name">{name}</span>'
            f'<span class="fe-bar"><span class="fe-fill" style="width:{pct:.1f}%"></span></span>'
            f'<span class="fe-pct">{pct:.1f}%</span>'
            f'</div>{wrap_close}'
        )
    return "".join(rows)


def render_status(predictor: Predictor, lang: str) -> str:
    if predictor.ready:
        color = "#3d7a38"
        msg = t(lang, "status_loaded", n=len(predictor.classes), path=predictor.ckpt_path)
    else:
        color = "#a0763a"
        msg = t(lang, "status_untrained")
    dot = (
        f'<span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{color};margin-right:8px;"></span>'
    )
    return (
        '<div style="font-family:Inter,Noto Sans JP,sans-serif;'
        f'color:#3a4a30;font-size:0.9rem;">{dot}{msg}</div>'
    )


# ---------- App ----------------------------------------------------------------

def build_app(ckpt_path: str | None) -> gr.Blocks:
    predictor = Predictor(ckpt_path)

    def analyze_fn(audio_path, top_k, lang, last_results):
        if audio_path is None:
            return render_results([], lang), []
        if not predictor.ready:
            return f'<div class="fe-empty">{t(lang, "empty_untrained")}</div>', []
        try:
            results = predictor.predict(audio_path, top_k=int(top_k))
        except Exception as e:  # noqa: BLE001
            err = html.escape(str(e))
            return f'<div class="fe-empty">{t(lang, "err_process", err=err)}</div>', []
        return render_results(results, lang), results

    def toggle_lang(current_lang, last_results):
        new_lang = "ja" if current_lang == "en" else "en"
        return (
            new_lang,
            hero_html(new_lang),
            footer_html(new_lang),
            gr.update(value=t(new_lang, "section_record")),
            gr.update(value=t(new_lang, "section_results")),
            gr.update(label=t(new_lang, "slider_top_k")),
            gr.update(value=t(new_lang, "btn_analyze")),
            gr.update(value=t(new_lang, "btn_lang")),
            render_status(predictor, new_lang),
            render_results(last_results or [], new_lang),
        )

    with gr.Blocks(title="ForestEcho") as app:
        lang_state = gr.State("en")
        last_results_state = gr.State([])

        with gr.Row(elem_classes="fe-lang-row"):
            lang_btn = gr.Button(t("en", "btn_lang"), size="sm", scale=0, min_width=110)

        hero = gr.HTML(hero_html("en"))

        with gr.Row():
            with gr.Column(scale=1, elem_classes="fe-card"):
                section_record = gr.Markdown(t("en", "section_record"))
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
                top_k = gr.Slider(1, 10, value=5, step=1, label=t("en", "slider_top_k"))
                analyze = gr.Button(t("en", "btn_analyze"), variant="primary", size="lg")
                status = gr.HTML(render_status(predictor, "en"))

            with gr.Column(scale=1, elem_classes="fe-card"):
                section_results = gr.Markdown(t("en", "section_results"))
                results_html = gr.HTML(render_results([], "en"))

        footer = gr.HTML(footer_html("en"))

        analyze.click(
            analyze_fn,
            inputs=[audio, top_k, lang_state, last_results_state],
            outputs=[results_html, last_results_state],
        )
        audio.change(
            analyze_fn,
            inputs=[audio, top_k, lang_state, last_results_state],
            outputs=[results_html, last_results_state],
        )
        lang_btn.click(
            toggle_lang,
            inputs=[lang_state, last_results_state],
            outputs=[
                lang_state, hero, footer,
                section_record, section_results,
                top_k, analyze, lang_btn,
                status, results_html,
            ],
        )

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

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
