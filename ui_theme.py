"""Theme CSS and escaped HTML fragments for the saved-result viewer."""

from html import escape
import math

from ui_helpers import DIMENSION_LABELS, weakest_dimension


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { --ink:#0F1020; --muted:#707184; --pink:#E6007E; --purple:#A83CF0; --line:#ECECF2; }
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; }
.stApp { background:#fff; color:var(--ink); }
.block-container { max-width:100%; padding:2.2rem 1.5rem 3rem; }
h1,h2,h3,p,div,span { font-family:'Plus Jakarta Sans',sans-serif; }
.vs-hero { position:relative; isolation:isolate; overflow:visible; padding:1.8rem 0 2.2rem; text-align:center; }
.vs-hero::before { content:""; position:absolute; z-index:-1; inset:-1rem -1.5rem -.5rem; pointer-events:none; filter:blur(42px); opacity:.3; background:radial-gradient(ellipse at 18% 42%,rgba(230,0,126,.28),transparent 42%),radial-gradient(ellipse at 78% 30%,rgba(168,60,240,.24),transparent 44%),radial-gradient(ellipse at 55% 82%,rgba(47,123,255,.17),transparent 40%); }
.vs-wordmark { font-size:.9rem; font-weight:800; letter-spacing:-.02em; color:var(--ink); margin-bottom:2.4rem; }
.vs-wordmark-mark { display:inline-block; width:9px; height:9px; margin-right:9px; border-radius:50%; background:linear-gradient(135deg,#E6007E,#A83CF0); }
.vs-headline { margin:0 auto; white-space:nowrap; font-size:clamp(26px,4.2vw,56px); line-height:1.12; letter-spacing:-.01em; word-spacing:normal; font-weight:800; color:#0F1020; }
.vs-headline-gradient { background:linear-gradient(100deg,#E6007E 0%,#A83CF0 75%); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.vs-subtitle { max-width:640px; margin:.85rem auto 1.2rem; color:var(--muted); font-size:1rem; }
.vs-button { display:inline-flex; border-radius:999px; background:linear-gradient(100deg,#E6007E,#A83CF0); color:#fff; padding:.7rem 1.25rem; font-size:.9rem; font-weight:700; opacity:.52; }
.vs-how-title { text-align:center; margin:.2rem 0 1rem; color:var(--ink); font-size:1rem; font-weight:800; }
.vs-feature { height:100%; min-height:178px; padding:19px; border:1px solid #EEEEF3; border-radius:20px; box-shadow:0 12px 36px rgba(15,16,32,.055); background:#fff; transition:transform .18s ease,box-shadow .18s ease; }
.vs-feature:hover { transform:translateY(-4px); box-shadow:0 18px 42px rgba(15,16,32,.11); }
.vs-feature-dark { background:#0F1020; border-color:#0F1020; color:#fff; }
.vs-feature-label { display:inline-block; border-radius:999px; padding:.28rem .62rem; color:#A545C9; background:#F8EAF8; font-size:.65rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
.vs-feature-dark .vs-feature-label { color:#fff; background:rgba(255,255,255,.14); }
.vs-feature-icon { font-size:1.35rem; margin:.8rem 0 .25rem; }
.vs-feature-title { font-weight:800; font-size:.95rem; margin:.2rem 0 .4rem; color:#0F1020; }
.vs-feature-dark .vs-feature-title { color:#fff; }
.vs-feature-copy { color:#707184; font-size:.76rem; line-height:1.5; }
.vs-feature-dark .vs-feature-copy { color:#D2D1DB; }
.vs-card { border:1px solid #EEEEF3; border-radius:20px; padding:24px; box-shadow:0 12px 36px rgba(15,16,32,.065); background:#fff; margin:.45rem 0 1.1rem; }
.vs-upload { background:#FCFBFD; }
.st-key-upload-panel { width:100%; max-width:820px; margin:0 auto; border-radius:20px; border-color:#EEEEF3; box-shadow:0 12px 36px rgba(15,16,32,.065); }
.st-key-upload-panel > div { padding:8px; }
[data-testid="stFileUploaderDropzone"] { box-sizing:border-box; min-height:210px; padding:1.5rem !important; border:2px dashed #DAD9E2 !important; border-radius:20px !important; background:#FCFBFD !important; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.55rem; transition:border-color .18s ease,background-color .18s ease; }
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploaderDropzone"]:focus-within { border-color:#E6007E !important; background:#FFF5FA !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { text-align:center; order:1; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content:""; display:block; width:42px; height:42px; margin:0 auto .35rem; border-radius:13px; background:#FCE5F2; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { content:"⇧"; position:absolute; margin:-2.65rem 0 0 -1.1rem; color:#E6007E; font-family:Arial,sans-serif; font-size:1.5rem; font-weight:700; }
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small { font-size:0 !important; line-height:0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span::before { content:"Drop your reel here or click to upload"; display:block; color:#29283A; font-family:'Plus Jakarta Sans',sans-serif; font-size:.92rem; font-weight:700; line-height:1.5; }
[data-testid="stFileUploaderDropzoneInstructions"] small::before { content:"MP4 or MOV, up to 100 MB"; display:block; margin-top:.25rem; color:#858493; font-family:'Plus Jakarta Sans',sans-serif; font-size:.76rem; line-height:1.5; }
[data-testid="stFileUploaderDropzone"] button { order:2; }
[data-testid="stFileUploaderDropzone"] button { min-height:0 !important; padding:.52rem 1rem !important; border:0 !important; border-radius:999px !important; background:#E6007E !important; color:#fff !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-size:.78rem !important; font-weight:700 !important; }
[data-testid="stFileUploaderDropzone"] button:hover { background:#C9006D !important; color:#fff !important; }
.st-key-upload-panel { background:linear-gradient(145deg,#fff 0%,#FFFAFD 55%,#FBF8FF 100%); border-color:#F0E8F1; border-radius:24px; box-shadow:0 16px 42px rgba(111,44,120,.09); }
[data-testid="stFileUploaderDropzone"] { min-height:200px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; text-align:center; }
[data-testid="stFileUploaderDropzone"] button { order:2; align-self:center; }
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] svg { display:none !important; }
[data-testid="stFileUploaderDropzone"] > div:has([data-testid="stBaseButton-secondary"]) { order:2; display:flex; justify-content:center; align-self:center; }
[data-testid="stFileUploaderDropzoneInstructions"] { order:1; position:relative; text-align:center; font-size:0 !important; line-height:0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div { visibility:hidden; height:0; overflow:hidden; font-size:0 !important; line-height:0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content:"Drop your reel here or click to upload"; display:block; visibility:visible; width:auto; height:auto; margin:0; border-radius:0; background:none; color:#29283A; font-family:'Plus Jakarta Sans',sans-serif; font-size:.95rem; font-weight:750; line-height:1.5; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { content:"MP4 or MOV, up to 100 MB"; position:static; display:block; visibility:visible; margin:.35rem 0 0; color:#858493; font-family:'Plus Jakarta Sans',sans-serif; font-size:.78rem; line-height:1.5; }
[data-testid="stFileUploaderDropzoneInstructions"] span::before, [data-testid="stFileUploaderDropzoneInstructions"] small::before { content:none !important; display:none !important; }
.vs-step-title { display:flex; align-items:center; gap:.65rem; margin:1.15rem .35rem .55rem; color:#29283A; font-size:.94rem; font-weight:800; }
.vs-step-title > span:first-child { display:inline-flex; width:27px; height:27px; flex:0 0 27px; align-items:center; justify-content:center; border-radius:50%; background:linear-gradient(135deg,#E6007E,#A83CF0); color:#fff; font-size:.72rem; box-shadow:0 4px 11px rgba(168,60,240,.18); }
.vs-context-title { margin-bottom:.55rem; }
.vs-step-optional { margin-left:-.25rem; padding:.2rem .55rem; border:1px solid #F0D7EB; border-radius:999px; background:#FFF5FB; color:#A34791; font-size:.62rem; font-weight:700; text-transform:lowercase; }
.st-key-upload-panel [data-testid="stTextAreaRootElement"] { border:0 !important; border-left:2px solid transparent !important; border-radius:16px !important; background:linear-gradient(#FFFAFD,#FFFAFD) padding-box,linear-gradient(180deg,#E6007E,#A83CF0) border-box !important; box-shadow:0 5px 18px rgba(168,60,240,.06); transition:box-shadow .18s ease; }
.st-key-upload-panel [data-testid="stTextAreaRootElement"]:focus-within { box-shadow:0 0 0 3px rgba(230,0,126,.10),0 0 18px rgba(168,60,240,.18) !important; }
.st-key-upload-panel [data-testid="stTextAreaRootElement"] textarea { border:0 !important; background:transparent !important; }
.vs-niche-hint { min-height:1.25rem; margin:.45rem .4rem .85rem; color:#777184; font-size:.76rem; line-height:1.45; }
.st-key-upload-panel [data-testid="stButtonGroup"] button { border-radius:999px !important; transition:filter .15s ease,box-shadow .15s ease; }
.st-key-upload-panel [data-testid="stButtonGroup"] > div:last-child, .st-key-upload-panel [data-testid="stButtonGroup"] [role="group"] { justify-content:center; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:hover { filter:brightness(.97); }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(1) { color:#B10068 !important; border-color:#EBC5DF !important; background:#FFF0F8 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(1)[data-selected] { color:#fff !important; border-color:transparent !important; background:linear-gradient(100deg,#E6007E,#A83CF0) !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(2) { color:#2767B4 !important; border-color:#C7DDF9 !important; background:#F0F7FF !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(3) { color:#B45A16 !important; border-color:#F6D2B2 !important; background:#FFF5EB !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(4) { color:#B52E70 !important; border-color:#F4C6DD !important; background:#FFF0F7 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(5) { color:#25805E !important; border-color:#BDE7D3 !important; background:#EFFBF5 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(6) { color:#7531B5 !important; border-color:#DEC9F4 !important; background:#F8F1FF !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(7) { color:#74717D !important; border-color:#E0DDE5 !important; background:#F7F6F9 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(8) { color:#A55A18 !important; border-color:#F2D3A8 !important; background:#FFF7E8 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(9) { color:#74717D !important; border-color:#E0DDE5 !important; background:#F7F6F9 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(n+2)[data-selected] { box-shadow:0 0 0 2px rgba(20,16,32,.09) inset; filter:saturate(1.12); }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(2)[data-selected] { color:#fff !important; background:#397EC8 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(3)[data-selected] { color:#fff !important; background:#D87327 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(4)[data-selected] { color:#fff !important; background:#D84588 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(5)[data-selected] { color:#fff !important; background:#2D966E !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(6)[data-selected] { color:#fff !important; background:#8645C2 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(7)[data-selected] { color:#fff !important; background:#77737F !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(8)[data-selected] { color:#fff !important; background:#C17A28 !important; }
.st-key-upload-panel [data-testid="stButtonGroup"] button:nth-of-type(9)[data-selected] { color:#fff !important; background:#77737F !important; }
.vs-card-title { color:var(--ink); font-size:1rem; font-weight:800; margin:0 0 .75rem; }
.vs-pill { display:inline-block; border-radius:999px; padding:.32rem .7rem; font-size:.7rem; line-height:1.2; font-weight:700; background:#F3F1F7; color:#656477; }
.vs-pill.active { color:#B10068; background:#FCE5F2; }
.vs-pills { display:flex; flex-wrap:wrap; gap:.45rem; margin:.85rem 0 .25rem; }
.vs-progress { display:flex; flex-wrap:wrap; gap:.55rem; margin:.1rem 0 1.2rem; }
.vs-progress .vs-pill.done { color:#126A47; background:#E8F7F0; }
.vs-progress .vs-pill.done::before { content:"✓"; margin-right:.35rem; }
.vs-progress .vs-pill.pending, .vs-progress .vs-pill.upcoming { color:#8A8997; background:#F3F3F6; }
.vs-progress .vs-pill.active, .vs-progress .vs-pill.current { color:#fff; background:linear-gradient(100deg,#E6007E,#A83CF0); box-shadow:0 0 0 0 rgba(168,60,240,.32); animation:vs-glow 2s ease-in-out infinite; }
.vs-progress .vs-pill.failed { color:#9B1C1C; background:#FDECEC; }
.vs-progress .vs-pill.skipped { color:#6B6578; background:#ECEAF0; }
@keyframes vs-glow { 50% { box-shadow:0 0 13px 3px rgba(168,60,240,.25); } }
.vs-wait-copy { margin:.35rem 0 .1rem; color:#4E4D60; font-weight:600; }
.vs-elapsed { color:#9694A2; font-size:.72rem; }
.vs-game-frame { width:100%; height:260px; border:0; }
.vs-tier { position:relative; border:0; color:#fff; padding:2rem 2rem 3.75rem; overflow:hidden; }
.vs-tier-low { background:linear-gradient(115deg,#6A4C93,#A83CF0); }
.vs-tier-medium { background:linear-gradient(115deg,#E6007E,#A83CF0); }
.vs-tier-high { background:linear-gradient(115deg,#A83CF0,#2F7BFF); }
.vs-tier-label { font-size:clamp(2.3rem,6vw,4rem); line-height:1; letter-spacing:-.06em; font-weight:800; margin:.7rem 0 1.2rem; }
.vs-tier-reason { max-width:740px; color:rgba(255,255,255,.86); font-size:.8rem; line-height:1.5; margin:.65rem 0 0; }
.vs-tier-kicker { font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; opacity:.82; }
.vs-estimate-title { font-size:.8rem; font-weight:700; margin:1.4rem 0 .55rem; }
.vs-estimate-values { display:flex; justify-content:space-between; gap:.75rem; font-size:.82rem; margin-top:.3rem; }
.vs-estimate-note { display:table; margin-top:.35rem; padding:.2rem .45rem; border-radius:6px; background:rgba(43,15,67,.48); color:#F5EFFA; font-size:.78rem; font-weight:600; line-height:1.45; }
.vs-estimate-confidence { position:absolute; right:2rem; bottom:1.15rem; padding:.28rem .62rem; border-radius:999px; background:rgba(43,15,67,.58); color:#fff; font-size:.76rem; font-weight:700; line-height:1.4; }
.vs-mini-card { min-height:155px; }
.vs-mini-value { font-size:1.55rem; font-weight:800; letter-spacing:-.04em; color:#242236; margin:.65rem 0 .4rem; }
.vs-mini-copy { color:var(--muted); font-size:.83rem; line-height:1.6; }
.vs-wider-label-line { display:flex; align-items:center; flex-wrap:wrap; gap:.55rem; margin:.65rem 0 .45rem; }
.vs-wider-label { color:#242236; font-size:1.55rem; line-height:1.1; letter-spacing:-.04em; font-weight:800; }
.vs-wider-pill { display:inline-block; padding:.25rem .65rem; border-radius:999px; font-size:.68rem; line-height:1.2; font-weight:800; }
.vs-wider-pill-narrow { color:#fff; background:linear-gradient(110deg,#6A4C93,#A83CF0); }
.vs-wider-pill-moderate { color:#A22A6F; background:#FCE5F2; }
.vs-wider-pill-broad { color:#fff; background:linear-gradient(100deg,#E6007E,#A83CF0); }
.vs-wider-unavailable { color:#858493; background:#F0EFF3; }
.vs-demo-banner { margin:.2rem 0 1rem; padding:.75rem 1rem; border:1px solid #E8C8F0; border-radius:14px; background:linear-gradient(100deg,#FFF0F8,#F5EDFF); color:#762C83; text-align:center; font-size:.85rem; font-weight:800; }
.vs-demo-tag { display:inline-block; margin:.55rem 0 0 .5rem; padding:.25rem .6rem; border:1px solid rgba(255,255,255,.38); border-radius:999px; background:rgba(255,255,255,.18); color:#fff; font-size:.66rem; font-weight:800; letter-spacing:.02em; vertical-align:middle; }
.vs-section-label { color:#777889; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; font-weight:800; margin:1.6rem 0 .65rem; }
.vs-heat-row { display:grid; grid-template-columns:minmax(110px,170px) 1fr 36px; gap:.75rem; align-items:center; margin:.8rem 0; }
.vs-heat-name { font-size:.78rem; color:#454456; font-weight:600; }
.vs-heat-track { height:8px; overflow:hidden; border-radius:999px; background:#F0EEF4; }
.vs-heat-value { text-align:right; font-size:.72rem; color:#777889; font-weight:700; }
.vs-dimension-row { display:grid; grid-template-columns:minmax(105px,135px) 1fr 38px auto; gap:.55rem; align-items:center; margin:.7rem 0; }
.vs-dimension-name { color:#454456; font-size:.75rem; font-weight:650; }
.vs-dimension-track { height:9px; overflow:hidden; border-radius:999px; background:#F0EEF4; }
.vs-dimension-value { color:#777889; font-size:.72rem; text-align:right; font-weight:750; }
.vs-weakest-tag { padding:.12rem .38rem; border-radius:999px; color:#9B347A; background:#FCE5F2; font-size:.58rem; font-weight:800; text-transform:uppercase; }
.vs-sound-warning { display:inline-block; margin:.25rem 0 .7rem; padding:.4rem .7rem; border:1px solid #F2C9DE; border-radius:999px; background:#FFF5FA; color:#9B347A; font-size:.72rem; font-weight:750; }
.vs-suggestion { min-height:210px; }
.vs-suggestion-text { font-size:.95rem; line-height:1.65; font-weight:600; color:#202033; margin:.85rem 0 1rem; }
.vs-quote { border-left:3px solid #E6007E; background:#FBF8FC; padding:.85rem 1rem; border-radius:0 12px 12px 0; color:#626174; font-size:.82rem; line-height:1.6; }
.vs-quote-person { display:block; margin-top:.45rem; font-size:.7rem; font-weight:700; color:#9694A2; }
.vs-chat-bubble { position:relative; min-height:146px; padding:17px 18px; margin:.35rem 0 .7rem; border:1px solid #EEEEF3; border-radius:18px; background:#fff; box-shadow:0 9px 28px rgba(15,16,32,.055); }
.vs-chat-head { display:flex; align-items:center; gap:.65rem; margin-bottom:.75rem; }
.vs-avatar { display:flex; width:36px; height:36px; flex:0 0 36px; align-items:center; justify-content:center; border-radius:50%; color:#fff; font-weight:800; font-size:.78rem; background:linear-gradient(135deg,#E6007E,#A83CF0); }
.vs-avatar-genz { background:linear-gradient(135deg,#E6007E,#A83CF0); }
.vs-avatar-casual { background:linear-gradient(135deg,#2F7BFF,#A83CF0); }
.vs-avatar-niche_fan { background:linear-gradient(135deg,#00A88F,#2F7BFF); }
.vs-avatar-skeptic { background:linear-gradient(135deg,#636477,#0F1020); }
.vs-avatar-creator { background:linear-gradient(135deg,#F47B20,#E6007E); }
.vs-avatar-marketer { background:linear-gradient(135deg,#A83CF0,#2F7BFF); }
.vs-avatar-suggestion { background:linear-gradient(135deg,#E6007E,#A83CF0); }
.vs-avatar-unknown { background:linear-gradient(135deg,#858594,#4B4B5B); }
.vs-chat-person { color:#29283A; font-size:.76rem; font-weight:800; overflow-wrap:anywhere; }
.vs-chat-badge { display:inline-block; margin-top:.16rem; color:#858493; font-size:.64rem; font-weight:700; }
.vs-chat-quote { margin:0; color:#5F5E70; font-size:.82rem; line-height:1.6; overflow-wrap:anywhere; }
.vs-audience-summary { margin:.5rem 0 1.1rem; padding:1rem 1.15rem; border:1px solid #F0E7F1; border-radius:18px; background:linear-gradient(110deg,#FFF8FC,#FBF8FF); }
.vs-audience-summary-lead { margin:0 0 .75rem; color:#5B3C5E; font-size:.82rem; font-weight:800; }
.vs-audience-counters { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; }
.vs-audience-counter { padding:.58rem .7rem; border-radius:12px; background:rgba(255,255,255,.78); }
.vs-audience-counter strong { display:block; color:#272239; font-size:1rem; }
.vs-audience-counter span { color:#777184; font-size:.7rem; }
.vs-wall-row-label { margin:.8rem 0 .4rem; color:#777184; font-size:.67rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.vs-audience-spotlight { margin:1rem 0 1.5rem; padding:1.1rem 1.2rem; border:1px solid #EDE4F0; border-radius:20px; background:linear-gradient(120deg,#fff 0%,#FFF8FC 48%,#F9F5FF 100%); box-shadow:0 12px 30px rgba(60,35,70,.07); }
.vs-spotlight-head { display:flex; align-items:center; gap:.75rem; }
.vs-spotlight-person { color:#29283A; font-size:.9rem; font-weight:800; overflow-wrap:anywhere; }
.vs-spotlight-archetype { margin-top:.15rem; color:#858493; font-size:.68rem; font-weight:600; }
.vs-spotlight-score { margin-left:auto; color:#9C267A; font-size:1.5rem; font-weight:800; white-space:nowrap; }
.vs-spotlight-score small { color:#8A8997; font-size:.65rem; font-weight:600; }
.vs-spotlight-quote { margin:.85rem 0 0; color:#555367; font-size:.87rem; line-height:1.6; overflow-wrap:anywhere; }
.vs-history-heading { display:flex; align-items:baseline; gap:.55rem; margin:1.6rem 0 .3rem; color:#171625; font-size:1.55rem; font-weight:800; letter-spacing:-.04em; }
.vs-history-count { color:#908D9D; font-size:.78rem; font-weight:700; letter-spacing:0; }
.vs-history-card { min-height:126px; margin:.25rem 0 .55rem; padding:.8rem .85rem; border:1px solid #ECE8F0; border-radius:16px; background:#fff; transition:border-color .15s ease,box-shadow .15s ease; }
.vs-history-card-selected { border:2px solid transparent; background:linear-gradient(#fff,#fff) padding-box,linear-gradient(110deg,#E6007E,#A83CF0) border-box; box-shadow:0 7px 20px rgba(168,60,240,.12); }
.vs-history-card-incomplete { border-color:#E1DFE6; background:#F7F7F9; color:#777582; }
.vs-history-name { margin:0 0 .55rem; color:#272438; font-size:.88rem; font-weight:800; overflow-wrap:anywhere; }
.vs-history-card-incomplete .vs-history-name { color:#777582; }
.vs-history-tier { display:inline-block; margin:0 .3rem .35rem 0; padding:.2rem .5rem; border-radius:999px; font-size:.62rem; font-weight:800; }
.vs-history-tier-low { color:#654B7F; background:#EEE8F6; }
.vs-history-tier-medium { color:#A22A6F; background:#FCE5F2; }
.vs-history-tier-high { color:#fff; background:linear-gradient(100deg,#A83CF0,#2F7BFF); }
.vs-history-tier-incomplete { color:#73717D; background:#E7E6EB; }
.vs-history-score { margin:.1rem 0 .4rem; color:#555367; font-size:.75rem; font-weight:700; }
.vs-history-meta { display:flex; flex-wrap:wrap; gap:.3rem .55rem; color:#858493; font-size:.66rem; line-height:1.45; }
.vs-history-niche { color:#9B347A; font-weight:700; }
.vs-history-placeholder { display:flex; height:104px; align-items:center; justify-content:center; margin-bottom:.45rem; border-radius:13px; background:linear-gradient(135deg,#FCE5F2,#EDE4FA 55%,#DCE7FF); color:#8C3F9D; font-size:2rem; font-weight:800; }
.st-key-history-tile button { width:100%; }
.st-key-history-rename button { min-width:34px; }
.vs-footer { text-align:center; border-top:1px solid var(--line); padding:1.2rem 0 .2rem; margin-top:2.4rem; color:#898896; font-size:.78rem; }
.st-key-upload-panel [data-testid="stPills"] { justify-content:center; }
.st-key-upload-panel [data-testid="stButton"] { display:flex; justify-content:center; }
.st-key-upload-panel [data-testid="stButton"] button { min-width:170px; }
@media(max-width:620px) {
  .block-container { padding:1.2rem 1rem 2rem; }
  .vs-wordmark { margin-bottom:1.8rem; }
  .vs-card { padding:19px; border-radius:17px; }
  .vs-tier { padding-bottom:3.5rem; }
  .vs-estimate-confidence { right:19px; bottom:1rem; }
  .vs-heat-row { grid-template-columns:100px 1fr 32px; gap:.5rem; }
  .vs-audience-counters { grid-template-columns:1fr; }
  .vs-audience-counter { display:flex; align-items:center; justify-content:space-between; gap:.75rem; }
}
@media(max-width:559px) {
  .vs-headline { white-space:normal; }
}
</style>
"""


def hero_html():
    return (
        '<section class="vs-hero"><div class="vs-wordmark"><span class="vs-wordmark-mark"></span>'
        'Virality Simulator</div><h1 class="vs-headline">See how your reel could land, '
        '<span class="vs-headline-gradient">before you post it.</span></h1>'
        '<p class="vs-subtitle">Get a clearer read on how different audiences might respond.</p>'
        '<span class="vs-button">Analyze a reel</span></section>'
    )


def demo_banner_html(enabled=True):
    """Return the fixed demo-mode notice only when environment gating is active."""
    if not enabled:
        return ""
    message = "DEMO MODE: no API calls. Results shown are replays of saved runs."
    return f'<div class="vs-demo-banner">{escape(message, quote=True)}</div>'


def feature_card_html(label, icon, title, copy, dark=False):
    dark_class = " vs-feature-dark" if dark else ""
    return (
        f'<section class="vs-feature{dark_class}"><span class="vs-feature-label">{escape(str(label), quote=True)}</span>'
        f'<div class="vs-feature-icon">{escape(str(icon), quote=True)}</div>'
        f'<div class="vs-feature-title">{escape(str(title), quote=True)}</div>'
        f'<div class="vs-feature-copy">{escape(str(copy), quote=True)}</div></section>'
    )


def chat_bubble_html(comment):
    archetype = comment.get("archetype_key", "unknown")
    allowed = {"genz", "casual", "niche_fan", "skeptic", "creator", "marketer", "suggestion"}
    avatar_class = archetype if archetype in allowed else "unknown"
    return (
        f'<article class="vs-chat-bubble"><div class="vs-chat-head">'
        f'<span class="vs-avatar vs-avatar-{avatar_class}">{escape(str(comment.get("initial", "")), quote=True)}</span>'
        f'<span><span class="vs-chat-person">{escape(str(comment.get("persona_id", "")), quote=True)}</span><br>'
        f'<span class="vs-chat-badge">{escape(str(comment.get("badge", "")), quote=True)}</span></span>'
        f'</div><p class="vs-chat-quote">{escape(str(comment.get("quote", "")), quote=True)}</p></article>'
    )


def niche_chip_html(label, active=False):
    class_name = "vs-pill active" if active else "vs-pill"
    return f'<span class="{class_name}">{escape(str(label), quote=True)}</span>'


def progress_html(states, active_index=None):
    normalized = []
    for index, (label, state) in enumerate(states):
        if isinstance(state, bool):
            state = "done" if state else "upcoming"
            if index == active_index:
                state = "current"
        normalized.append((label, state))
    return '<div class="vs-progress">' + "".join(
        f'<span class="vs-pill {escape(str(state), quote=True)}">{escape(str(label), quote=True)}</span>'
        for label, state in normalized
    ) + '</div>'


def audience_tile_style_html(widget_key, bucket, badges=(), selected=False):
    """Return keyed CSS for a score-bucket audience tile and its badges."""
    import re

    if not re.fullmatch(r"audience_tile_[A-Za-z0-9_-]+", str(widget_key)):
        raise ValueError("Invalid audience tile widget key")
    colors = {
        "low": ("#F0EFF3", "#666574", "#D8D6DF"),
        "mid_low": ("#FCE5F2", "#9B376F", "#F2BED9"),
        "mid_high": ("#ED9BCB", "#59264F", "#E77BB9"),
        "high": ("linear-gradient(135deg,#E6007E,#A83CF0)", "#fff", "#A83CF0"),
    }
    background, foreground, border = colors.get(bucket, colors["low"])
    badge_values = [badge for badge in badges if badge in {"top", "low"}]
    badge_text = " / ".join(badge_values)
    badge_markup = ""
    if badge_text:
        badge_markup = (
            f'<style>.st-key-{widget_key} button::after{{content:"{escape(badge_text, quote=True)}";'
            'position:absolute;top:-5px;right:-4px;padding:1px 4px;border-radius:999px;'
            'background:#fff;color:#6B5273;font-size:8px;font-weight:800;line-height:1.3;'
            'box-shadow:0 1px 5px rgba(40,20,50,.15)}}</style>'
        )
    selected_css = " box-shadow:0 0 0 2px #342043,0 5px 12px rgba(90,35,100,.16);" if selected else ""
    return (
        f'<style>.st-key-{widget_key} button{{position:relative;background:{background} !important;'
        f'color:{foreground} !important;border:1px solid {border} !important;min-height:44px;'
        'padding:.32rem .3rem !important;font-size:.72rem !important;font-weight:800 !important;'
        f'{selected_css}'
        'white-space:nowrap;transition:transform .15s ease,box-shadow .15s ease}}'
        f'.st-key-{widget_key} button:hover{{transform:translateY(-2px);box-shadow:0 5px 12px rgba(90,35,100,.14)}}'
        f'</style>{badge_markup}'
    )


def history_tile_html(item, selected=False):
    """Build an escaped History tile from result and sidecar display fields."""
    display_name = escape(str(item.get("display_name", "Run")), quote=True)
    niche = escape(str(item.get("niche") or "Niche not recorded"), quote=True)
    date_label = escape(str(item.get("date_label", "Date unavailable")), quote=True)
    incomplete = bool(item.get("incomplete"))
    selected_class = " vs-history-card-selected" if selected else ""
    incomplete_class = " vs-history-card-incomplete" if incomplete else ""
    if incomplete:
        tier_markup = '<span class="vs-history-tier vs-history-tier-incomplete">Incomplete</span>'
        range_markup = ""
    else:
        tier = str(item.get("tier", "")).strip().lower()
        tier = tier.replace("tier", "").strip()
        tier_label = {"low": "Low", "medium": "Medium", "high": "High"}.get(tier, "Tier unavailable")
        tier_class = {"low": "low", "medium": "medium", "high": "high"}.get(tier, "low")
        tier_markup = (
            f'<span class="vs-history-tier vs-history-tier-{tier_class}">{escape(tier_label, quote=True)}</span>'
        )
        value = item.get("core_score")
        range_markup = ""
        if value is not None:
            try:
                score = float(value)
                confidence = str(item.get("confidence") or "Unavailable")
                range_markup = (
                    f'<div class="vs-history-score">{escape(f"{score:.1f} / 10 · Confidence: {confidence}", quote=True)}</div>'
                )
            except (TypeError, ValueError):
                pass
    return (
        f'<article class="vs-history-card{selected_class}{incomplete_class}">'
        f'<h3 class="vs-history-name">{display_name}</h3>{tier_markup}{range_markup}'
        f'<div class="vs-history-meta"><span class="vs-history-niche">{niche}</span>'
        f'<span>{date_label}</span></div></article>'
    )


def history_placeholder_html(display_name):
    """Return an escaped gradient placeholder when no local thumbnail exists."""
    initial = str(display_name or "R").strip()[:1].upper() or "R"
    return f'<div class="vs-history-placeholder">{escape(initial, quote=True)}</div>'


def result_hero_html(tier, niche, core_value, niche_reason="", demo_replay=False,
                     confidence=None, score_range=None):
    tier_text = str(tier)
    tier_class = {"low": "vs-tier-low", "medium": "vs-tier-medium", "high": "vs-tier-high"}.get(
        tier_text.lower(), "vs-tier-medium"
    )
    range_markup = ""
    if core_value is not None:
        score_text = "Unavailable"
        if (isinstance(score_range, (list, tuple)) and len(score_range) == 2
                and all(isinstance(bound, (int, float)) for bound in score_range)):
            score_text = f"{float(score_range[0]):.1f}–{float(score_range[1]):.1f}"
        confidence_label = escape(str(confidence or "Unavailable"), quote=True)
        range_markup = (
            '<div class="vs-estimate-title">Core audience estimate</div>'
            f'<div class="vs-estimate-values"><span>{escape(score_text, quote=True)}</span></div>'
            '<div class="vs-estimate-note">Estimated score is out of 10</div>'
            f'<div class="vs-estimate-confidence">Confidence: {confidence_label}</div>'
        )
    reason_label = "Niche set by you" if niche_reason == "user override" else "Auto-detected"
    reason_markup = f'<p class="vs-tier-reason">{escape(reason_label, quote=True)}</p>'
    demo_tag = '<span class="vs-demo-tag">Demo replay</span>' if demo_replay else ""
    return (
        f'<section class="vs-card vs-tier {tier_class}"><div class="vs-tier-kicker">Reach tier</div>'
        f'<div class="vs-tier-label">{escape(tier_text, quote=True)}{demo_tag}</div>'
        f'<span class="vs-pill">{escape(str(niche), quote=True)}</span>'
        f'{reason_markup}{range_markup}</section>'
    )


def score_card_html(title, value, sentence, confidence=None, score_range=None):
    score_text = "Unavailable"
    if (isinstance(score_range, (list, tuple)) and len(score_range) == 2
            and all(isinstance(bound, (int, float)) for bound in score_range)):
        score_text = f"{float(score_range[0]):.1f}–{float(score_range[1]):.1f}"
    confidence_text = escape(str(confidence or "Unavailable"), quote=True)
    return (
        f'<section class="vs-card vs-mini-card"><div class="vs-card-title">{escape(str(title), quote=True)}</div>'
        f'<div class="vs-mini-value">{escape(score_text, quote=True)}</div>'
        f'<div class="vs-mini-copy">Confidence: {confidence_text}</div>'
        f'<div class="vs-mini-copy">{escape(str(sentence), quote=True)}</div>'
        f'<div class="vs-mini-copy">95% percentile-bootstrap interval; requires at least 3 scored personas.</div></section>'
    )


def wider_audience_card_html(label):
    """Render the qualitative wider-audience card without exposing its raw score."""
    options = {
        "Narrow": ("narrow", "Mostly speaks to its own audience. Normal for niche or B2B reels."),
        "Moderate": ("moderate", "Could travel a little beyond its closest audience."),
        "Broad": ("broad", "Likely to appeal beyond its closest audience."),
    }
    if label not in options:
        label_text, badge_class, explanation = "Unavailable", "unavailable", ""
    else:
        badge_class, explanation = options[label]
        label_text = label
    return (
        '<section class="vs-card vs-mini-card"><div class="vs-card-title">Wider audience</div>'
        '<div class="vs-wider-label-line">'
        f'<div class="vs-wider-label">{escape(label_text, quote=True)}</div>'
        f'<span class="vs-wider-pill vs-wider-pill-{escape(badge_class, quote=True)}">'
        f'{escape(label_text, quote=True)}</span></div>'
        f'<div class="vs-mini-copy">{escape(explanation, quote=True)}</div></section>'
    )


def wider_audience_raw_score_html(raw_score):
    """Render the exact file-sourced crossover value for the Details expander."""
    value = "Unavailable" if raw_score is None else escape(str(raw_score), quote=True)
    return f'<p><strong>Wider audience (crossover) raw score:</strong> {value}</p>'


def dimension_bars_html(dimension_means, wave, condition, weakest=False):
    """Render escaped, bounded score bars from saved dimension means."""
    wave_data = dimension_means.get(wave) if isinstance(dimension_means, dict) else None
    values = wave_data.get(condition) if isinstance(wave_data, dict) else None
    values = values if isinstance(values, dict) else {}
    weakest_key = weakest_dimension(dimension_means, wave, condition) if weakest else None
    rows = []
    for key, label in DIMENSION_LABELS.items():
        raw = values.get(key)
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
            bounded = min(10.0, max(0.0, value))
            displayed = f"{value:.1f}"
        except (TypeError, ValueError):
            bounded, displayed = 0.0, "—"
        weakest_tag = '<span class="vs-weakest-tag">weakest</span>' if key == weakest_key else ""
        fill_width = f"{bounded * 10:g}%"
        fill_style = (
            f"display:block;width:{fill_width};height:100%;min-width:{'3px' if bounded > 0 else '0'};"
            "border-radius:inherit;background-image:linear-gradient(90deg,#E6007E,#A83CF0);"
        )
        rows.append(
            '<div class="vs-dimension-row">'
            f'<span class="vs-dimension-name">{escape(label, quote=True)}</span>'
            '<div class="vs-dimension-track" style="width:100%;height:8px;position:relative;'
            'background:#F0EEF4;border-radius:999px;overflow:hidden"><div style="'
            f'{fill_style}"></div></div>'
            f'<span class="vs-dimension-value">{escape(displayed, quote=True)}</span>{weakest_tag}</div>'
        )
    return '<div class="vs-dimension-bars">' + "".join(rows) + "</div>"


def sound_warning_html(gap):
    """Render the Wave 1 muted-audio warning with a one-decimal gap."""
    try:
        points = float(gap)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(points) or points < 1.0:
        return ""
    message = f"Loses {points:.1f} points when muted. Add captions or on-screen text."
    return f'<div class="vs-sound-warning">{escape(message, quote=True)}</div>'


def heat_bar_html(label, value):
    if value is None:
        fill = 0.0
        displayed = "—"
    else:
        fill = min(10.0, max(0.0, float(value))) * 10
        displayed = f"{float(value):.1f}"
    fill_width = f"{fill:g}%"
    fill_style = (
        f"display:block;width:{fill_width};height:100%;min-width:{'3px' if fill > 0 else '0'};"
        "border-radius:inherit;background-image:linear-gradient(90deg,#E6007E,#A83CF0);"
    )
    return (
        '<div class="vs-heat-row">'
        f'<span class="vs-heat-name">{escape(str(label), quote=True)}</span>'
        '<div class="vs-heat-track" style="width:100%;height:8px;position:relative;'
        'background:#F0EEF4;border-radius:999px;overflow:hidden"><div style="'
        f'{fill_style}"></div></div>'
        f'<span class="vs-heat-value">{displayed}</span></div>'
    )


def suggestion_card_html(criterion, suggestion, quote="", persona_id=""):
    quote_block = ""
    if quote:
        quote_block = (
            f'<blockquote class="vs-quote">{escape(str(quote), quote=True)}'
            f'<span class="vs-quote-person">{escape(str(persona_id), quote=True)}</span></blockquote>'
        )
    return (
        '<section class="vs-card vs-suggestion">'
        f'<span class="vs-pill active">{escape(str(criterion or "Suggestion"), quote=True)}</span>'
        f'<div class="vs-suggestion-text">{escape(str(suggestion), quote=True)}</div>{quote_block}</section>'
    )
