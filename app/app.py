import gradio as gr
from PIL import Image
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import load_model, predict as run_predict, get_gradcam_tensor
from gradcam import GradCAM, overlay_heatmap
from utils import load_knowledge_base, get_car_info

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
MODEL, CLASS_NAMES, DEVICE = load_model(MODEL_PATH)
GRADCAM = GradCAM(MODEL)
KB      = load_knowledge_base(os.path.join(os.path.dirname(__file__), '..', 'data', 'car_knowledge.json'))

def analyse_car(image):
    if image is None:
        empty = '<div style="color:#333;font-size:14px;text-align:center;padding:60px 0">Upload a photo and click Identify</div>'
        return None, empty, empty

    pil_img    = Image.fromarray(image).convert('RGB')
    results    = run_predict(pil_img, MODEL, CLASS_NAMES, DEVICE)
    top_label, top_conf = results[0]
    info       = get_car_info(top_label, KB)
    low_conf   = top_conf < 60.0

    tensor  = get_gradcam_tensor(pil_img, DEVICE)
    heatmap = GRADCAM.generate(tensor, CLASS_NAMES.index(top_label))
    overlay = overlay_heatmap(np.array(pil_img), heatmap)

    pred_html = '<div style="font-family:\'Bebas Neue\',sans-serif;letter-spacing:1px;font-size:13px;color:#F5620F;margin-bottom:14px;text-transform:uppercase">Top Predictions</div>'
    for i, (label, conf) in enumerate(results):
        bar_color = '#22C55E' if conf > 80 else '#F5620F' if conf > 50 else '#555550'
        medal     = ['1.','2.','3.','4.','5.'][i]
        width     = max(conf, 2)
        pred_html += f'''
        <div style="background:#1a1a1a;border-radius:10px;padding:14px 16px;margin-bottom:8px;border:1px solid #222;border-left:3px solid {bar_color}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="color:#F0EDE8;font-size:14px;font-weight:600">{medal} {label}</span>
                <span style="color:{bar_color};font-weight:700;font-family:\'DM Mono\',monospace;font-size:14px">{conf:.1f}%</span>
            </div>
            <div style="background:#2a2a2a;border-radius:3px;height:4px">
                <div style="background:{bar_color};width:{width:.1f}%;height:4px;border-radius:3px"></div>
            </div>
        </div>'''

    if low_conf:
        pred_html += '<div style="background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.2);border-radius:10px;padding:12px 16px;margin-top:8px"><span style="color:#F5A623;font-size:13px">Low confidence — try a clearer photo with good lighting</span></div>'

    score   = info.get('reliability_score', 'N/A')
    cost    = info.get('avg_repair_cost_annual', 'N/A')
    price   = info.get('typical_price_range', 'N/A')
    verdict = info.get('verdict', '')
    issues  = info.get('common_issues', [])
    checks  = info.get('what_to_check', [])
    reddit  = info.get('reddit_communities', [])
    exps    = info.get('real_experiences', [])

    score_color  = '#22C55E' if isinstance(score, (int,float)) and score >= 8 else '#F5A623' if isinstance(score, (int,float)) and score >= 6 else '#EF4444'
    cost_display = f'${cost:,}/yr' if isinstance(cost, (int,float)) else str(cost)

    issues_html = ''.join([f'<div style="display:flex;gap:10px;margin-bottom:10px"><span style="color:#EF4444;flex-shrink:0">></span><span style="color:#888880;font-size:13px;line-height:1.5">{i}</span></div>' for i in issues])
    checks_html = ''.join([f'<div style="display:flex;gap:10px;margin-bottom:10px"><span style="color:#F5620F;flex-shrink:0">-></span><span style="color:#888880;font-size:13px;line-height:1.5">{c}</span></div>' for c in checks])

    reddit_html = ''.join([
        f'<a href="https://www.reddit.com/{sub.lstrip("/")}" target="_blank" '
        f'style="text-decoration:none;display:inline-block;background:#1a1a1a;border:1px solid #333;'
        f'border-radius:999px;padding:5px 12px;margin:0 6px 6px 0;color:#F5620F;font-size:12px;'
        f'font-family:\'DM Mono\',monospace">{sub}</a>'
        for sub in reddit]) or '<span style="color:#555;font-size:13px">No communities listed.</span>'
    exps_html = ''.join([f'<div style="display:flex;gap:10px;margin-bottom:10px"><span style="color:#22C55E;flex-shrink:0">"</span><span style="color:#888880;font-size:13px;line-height:1.5">{e}</span></div>' for e in exps])

    info_html = f'''
    <div style="font-family:\'Bebas Neue\',sans-serif;letter-spacing:1px;font-size:13px;color:#F5620F;margin-bottom:16px;text-transform:uppercase">Vehicle Profile — {top_label}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:20px">
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:14px;text-align:center">
            <div style="font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Reliability</div>
            <div style="font-family:\'Bebas Neue\',sans-serif;font-size:28px;color:{score_color}">{score}<span style="font-size:14px;color:#555">/10</span></div>
        </div>
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:14px;text-align:center">
            <div style="font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Annual Cost</div>
            <div style="font-family:\'Bebas Neue\',sans-serif;font-size:28px;color:#F5620F">{cost_display}</div>
        </div>
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:14px;text-align:center">
            <div style="font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Price Range</div>
            <div style="font-family:\'Bebas Neue\',sans-serif;font-size:18px;color:#F0EDE8;margin-top:4px">{price}</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:16px">
            <div style="font-size:10px;color:#EF4444;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Known Issues</div>
            {issues_html}
        </div>
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:16px">
            <div style="font-size:10px;color:#F5620F;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">What to Inspect</div>
            {checks_html}
        </div>
    </div>
    <div style="background:rgba(245,98,15,0.06);border:1px solid rgba(245,98,15,0.15);border-radius:10px;padding:16px;margin-bottom:16px">
        <div style="font-size:10px;color:#F5620F;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">Verdict</div>
        <div style="color:#F0EDE8;font-size:14px;line-height:1.6">{verdict}</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:16px">
            <div style="font-size:10px;color:#22C55E;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Owner Experiences</div>
            {exps_html}
        </div>
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:16px">
            <div style="font-size:10px;color:#F5620F;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Reddit Communities</div>
            {reddit_html}
        </div>
    </div>'''

    return overlay, pred_html, info_html


css = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
body, .gradio-container, .main { background: #080808 !important; font-family: 'DM Sans', sans-serif !important; }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding: 0 24px 60px !important; }
button.primary, .gr-button-primary { background: #F5620F !important; border-radius: 10px !important; font-weight: 700 !important; color: white !important; }
button.primary:hover { background: #c44d0c !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, title='CarIQ — Vehicle Intelligence') as demo:

    gr.HTML("""
    <div style="padding:40px 0 32px;border-bottom:1px solid #1a1a1a;margin-bottom:32px">
        <div style="display:flex;align-items:baseline;gap:2px;margin-bottom:10px">
            <span style="font-family:'Bebas Neue',sans-serif;font-size:42px;letter-spacing:4px;color:#F5620F">Car</span>
            <span style="font-family:'Bebas Neue',sans-serif;font-size:42px;letter-spacing:4px;color:#F0EDE8">IQ</span>
            <span style="font-family:'DM Mono',monospace;font-size:11px;color:#444;letter-spacing:2px;margin-left:12px;text-transform:uppercase;align-self:center">// Vehicle Intelligence</span>
        </div>
        <p style="color:#555;font-size:14px;margin:0;max-width:500px">Upload any car photo — get instant identification, reliability data, common issues, and repair costs.</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            gr.HTML('<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#F5620F;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px">Upload Photo</div>')
            img_input = gr.Image(label='', type='numpy', height=300, show_label=False)
            btn       = gr.Button('Identify Vehicle', variant='primary', size='lg')

        with gr.Column(scale=6):
            gr.HTML('<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#F5620F;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px">Results</div>')
            pred_output = gr.HTML(value='<div style="color:#333;text-align:center;padding:60px 0">Upload a photo and click Identify</div>', show_label=False)

    gr.HTML('<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#F5620F;letter-spacing:3px;text-transform:uppercase;margin:24px 0 10px">Grad-CAM — What the AI Saw</div>')
    gradcam_output = gr.Image(label='', show_label=False, height=250)

    gr.HTML('<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#F5620F;letter-spacing:3px;text-transform:uppercase;margin:24px 0 10px">Vehicle Profile</div>')
    info_output = gr.HTML(value='<div style="color:#333;text-align:center;padding:40px 0">Identification results will appear here</div>', show_label=False)

    gr.HTML("""
    <div style="border-top:1px solid #1a1a1a;margin-top:40px;padding-top:24px;display:flex;justify-content:space-between">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:18px;letter-spacing:3px;color:#F5620F">CarIQ</span>
        <span style="font-family:'DM Mono',monospace;font-size:11px;color:#333">Built by Marvin Ayala</span>
    </div>
    """)

    btn.click(fn=analyse_car, inputs=img_input, outputs=[gradcam_output, pred_output, info_output])

if __name__ == '__main__':
    demo.launch()
