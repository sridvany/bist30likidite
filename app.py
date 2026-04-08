import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# ── BIST30 Listesi ───────────────────────────────────────────────────────────
# Son güncelleme: Nisan 2025
BIST30 = [
    "AEFES.IS","AKBNK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS",
    "ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS",
    "ISCTR.IS","KCHOL.IS","KOZAL.IS","KRDMD.IS","MGROS.IS",
    "PETKM.IS","SAHOL.IS","SASA.IS","SISE.IS","TAVHL.IS",
    "TCELL.IS","THYAO.IS","TOASO.IS","TTKOM.IS","TUPRS.IS",
    "VAKBN.IS","YKBNK.IS","PGSUS.IS","ASTOR.IS",
]

# ── Sayfa Ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST30 Likidite Analizi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
}
.metric-box {
    background: #0f1117;
    border: 1px solid #2a2d3e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.ticker-badge {
    font-family: 'IBM Plex Mono', monospace;
    background: #1e2235;
    color: #7dd3fc;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: 600;
}
.oldest-date {
    font-size: 0.75em;
    color: #6b7280;
    font-family: 'IBM Plex Mono', monospace;
}
.pos { color: #22c55e; font-weight: 600; }
.neg { color: #ef4444; font-weight: 600; }
.neutral { color: #94a3b8; }
.winner-banner {
    background: linear-gradient(135deg, #1e2235 0%, #0f1117 100%);
    border: 1px solid #22c55e;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────
def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

@st.cache_data(ttl=60, show_spinner=False, persist=False)
def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = _flatten(df)
    df = df[~df.index.duplicated(keep="first")]
    df.dropna(subset=["Close", "Open", "High", "Low", "Volume"], inplace=True)
    return df

@st.cache_data(ttl=30, show_spinner=False, persist=False)
def fetch_live(ticker: str) -> pd.Series | None:
    try:
        intra = yf.download(ticker, period="1d", interval="1m",
                            auto_adjust=True, progress=False)
        if intra.empty:
            return None
        intra = _flatten(intra)
        today = date.today()
        today_ts = pd.Timestamp(today)
        row = pd.Series({
            "Open":   intra["Open"].iloc[0],
            "High":   intra["High"].max(),
            "Low":    intra["Low"].min(),
            "Close":  intra["Close"].iloc[-1],
            "Volume": intra["Volume"].sum(),
        }, name=today_ts)
        return row
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False, persist=False)
def fetch_oldest_date(ticker: str) -> str:
    try:
        df = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
        if df.empty:
            return "—"
        return df.index.min().strftime("%d.%m.%Y")
    except Exception:
        return "—"

@st.cache_data(ttl=120, show_spinner=False, persist=False)
def fetch_intraday(ticker: str, selected_date: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="60d", interval="2m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = _flatten(df)
        df.index = df.index.tz_convert("Europe/Istanbul")
        df = df[~df.index.duplicated(keep="first")]
        df.dropna(subset=["Close", "Open", "High", "Low", "Volume"], inplace=True)
        day_df = df[df.index.date == pd.Timestamp(selected_date).date()]
        return day_df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False, persist=False)
def fetch_intraday_60d(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="60d", interval="2m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = _flatten(df)
        df.index = df.index.tz_convert("Europe/Istanbul")
        df = df[~df.index.duplicated(keep="first")]
        df.dropna(subset=["Close", "Open", "High", "Low", "Volume"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# ── BIST30 Tarama Fonksiyonu ─────────────────────────────────────────────────
def fetch_bist30_best(baslangic_tarihi):
    """
    BIST30 hisselerini tara, belirtilen tarihten bugüne performansa göre en iyi hisseyi seç.
    Skor = 0.5 × getiri_rank + 0.5 × hacim_rank (her ikisi 0-1 normalize)
    """
    baslangic = str(baslangic_tarihi)

    sonuclar = []
    hatalar  = []

    # Progress UI
    st.markdown("#### 🔍 BIST30 Taranıyor...")
    prog_bar  = st.progress(0)
    durum_txt = st.empty()

    for i, ticker in enumerate(BIST30):
        durum_txt.markdown(
            f"<span style='font-family:IBM Plex Mono;font-size:0.85em;color:#94a3b8'>"
            f"⏳ **{ticker}** işleniyor... ({i+1}/{len(BIST30)})<br>"
            f"Yapılan işlem: {baslangic} tarihinden bugüne kapanış fiyatı ve hacim verisi çekiliyor, "
            f"getiri ve ortalama hacim hesaplanıyor.</span>",
            unsafe_allow_html=True
        )
        prog_bar.progress((i + 1) / len(BIST30))

        try:
            df = yf.download(ticker, start=baslangic, auto_adjust=True, progress=False)
            if df.empty:
                hatalar.append(ticker)
                continue
            df = _flatten(df)
            df.dropna(subset=["Close", "Volume"], inplace=True)

            df_n = df
            if len(df_n) < 2:
                hatalar.append(ticker)
                continue

            getiri     = (df_n["Close"].iloc[-1] / df_n["Close"].iloc[0]) - 1
            ort_hacim  = df_n["Volume"].mean()
            son_fiyat  = df_n["Close"].iloc[-1]

            sonuclar.append({
                "ticker":    ticker,
                "getiri":    getiri,
                "ort_hacim": ort_hacim,
                "son_fiyat": son_fiyat,
            })
        except Exception:
            hatalar.append(ticker)
            continue

    prog_bar.empty()
    durum_txt.empty()

    if not sonuclar:
        return None

    df_s = pd.DataFrame(sonuclar)

    # 0-1 normalize rank
    df_s["getiri_rank"] = (df_s["getiri"].rank() - 1) / (len(df_s) - 1)
    df_s["skor"]        = df_s["getiri_rank"]

    kazanan = df_s.sort_values("skor", ascending=False).iloc[0]

    return {
        "ticker":    kazanan["ticker"],
        "getiri":    kazanan["getiri"],
        "ort_hacim": kazanan["ort_hacim"],
        "skor":      kazanan["skor"],
        "getiri_rank": kazanan["getiri_rank"],
        "df_tum":    df_s.sort_values("skor", ascending=False).reset_index(drop=True),
        "hatalar":   hatalar,
    }


def compute_intraday_metrics(df: pd.DataFrame, df_60d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Kapanış"]        = df["Close"].round(4)
    out["Açılış"]         = df["Open"].round(4)
    out["Yüksek"]         = df["High"].round(4)
    out["Düşük"]          = df["Low"].round(4)
    out["Hacim"]          = df["Volume"].astype(int)
    out["Değişim (%)"]    = df["Close"].pct_change() * 100
    out["Bar Range (%)"]  = ((df["High"] - df["Low"]) / df["Low"] * 100).round(4)

    bar_return = df["Close"].pct_change().abs()
    tl_vol     = df["Close"] * df["Volume"]
    out["Amihud (2dk)"]   = (bar_return / tl_vol * 1e6).replace([np.inf, -np.inf], np.nan)

    h = np.log(df["High"])
    l = np.log(df["Low"])
    h2 = np.log(df["High"].combine(df["High"].shift(-1), max))
    l2 = np.log(df["Low"].combine(df["Low"].shift(-1), min))
    beta  = (h - l) ** 2 + (h.shift(-1) - l.shift(-1)) ** 2
    gamma = (h2 - l2) ** 2
    k     = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0)
    cs    = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    out["C-S Spread (%)"] = (cs * 100).round(4)

    if not df_60d.empty:
        df_60d = df_60d.copy()
        df_60d["time_key"] = df_60d.index.strftime("%H:%M")
        avg_vol = df_60d.groupby("time_key")["Volume"].mean()
        time_keys = df.index.strftime("%H:%M")
        rvol_vals = []
        for tk, v in zip(time_keys, df["Volume"]):
            avg = avg_vol.get(tk, np.nan)
            rvol_vals.append(round(v / avg, 3) if avg and avg > 0 else np.nan)
        out["RVOL"] = rvol_vals
    else:
        out["RVOL"] = np.nan

    return out


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Kapanış (₺)"]     = df["Close"].round(2)
    out["Açılış (₺)"]      = df["Open"].round(2)
    out["Yüksek (₺)"]      = df["High"].round(2)
    out["Düşük (₺)"]       = df["Low"].round(2)
    out["Hacim"]           = df["Volume"].astype(int)

    out["Günlük Değ. (%)"] = df["Close"].pct_change() * 100
    out["Güniçi Değ. (%)"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
    out["Daily Range (₺)"] = (df["High"] - df["Low"]).round(2)
    out["Daily Range (%)"] = ((df["High"] - df["Low"]) / df["Low"] * 100).round(2)

    tl_volume = df["Close"] * df["Volume"]
    daily_return = df["Close"].pct_change().abs()
    out["Amihud (×10⁶)"] = (daily_return / tl_volume * 1e6).replace([np.inf, -np.inf], np.nan)
    out["log₁₀(Hacim)"] = np.log10(df["Volume"].replace(0, np.nan))

    h = np.log(df["High"])
    l = np.log(df["Low"])
    h2 = np.log(df["High"].combine(df["High"].shift(-1), max))
    l2 = np.log(df["Low"].combine(df["Low"].shift(-1), min))
    beta  = (h - l) ** 2 + (h.shift(-1) - l.shift(-1)) ** 2
    gamma = (h2 - l2) ** 2
    k     = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0)
    cs    = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    out["C-S Spread (%)"] = (cs * 100).round(4)

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    window  = 90
    mec_vals = []
    for i in range(len(df)):
        if i < window:
            mec_vals.append(np.nan)
            continue
        seg = log_ret.iloc[i - window + 1: i + 1]
        lr30 = np.log(df["Close"].iloc[i - window + 1: i + 1].values[::6])
        r30  = np.diff(lr30)
        r5   = seg.values
        var30 = np.var(r30, ddof=1) if len(r30) > 1 else np.nan
        var5  = np.var(r5,  ddof=1) if len(r5)  > 1 else np.nan
        denom = 6 * var5
        mec_vals.append(round(var30 / denom, 4) if denom and denom > 0 else np.nan)
    out["MEC"] = mec_vals

    amihud    = out["Amihud (×10⁶)"].copy()
    log_hacim = out["log₁₀(Hacim)"].copy()
    out = out.round(4)
    out["Amihud (×10⁶)"]  = amihud
    out["log₁₀(Hacim)"]   = log_hacim.round(4)
    return out

def color_val(val, col):
    if pd.isna(val):
        return '<span class="neutral">—</span>'
    if col in ["Günlük Değ. (%)", "Güniçi Değ. (%)"]:
        cls = "pos" if val > 0 else ("neg" if val < 0 else "neutral")
        sign = "+" if val > 0 else ""
        return f'<span class="{cls}">{sign}{val:.2f}%</span>'
    if col == "C-S Spread (%)":
        return f'<span class="neutral">{val:.4f}%</span>'
    if col == "MEC":
        cls = "pos" if val <= 1.0 else "neg"
        return f'<span class="{cls}">{val:.4f}</span>'
    if col == "log₁₀(Hacim)":
        return f'<span class="neutral">{val:.4f}</span>'
    if col == "Amihud (×10⁶)":
        log_val = abs(np.log10(val)) if val > 0 else np.nan
        if np.isnan(log_val):
            return '<span class="neutral">—</span>'
        return f'<span class="neutral">{log_val:.2f}</span>'
    if col == "Daily Range (₺)":
        return f'<span class="neutral">{val:.2f}</span>'
    if col == "Daily Range (%)":
        return f'<span class="neutral">{val:.2f}%</span>'
    if col == "Hacim":
        return f'<span class="neutral">{int(val):,}</span>'
    return f'<span class="neutral">{val:,.2f}</span>'

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 BIST30 Likidite Analizi")
    st.markdown("---")

    analiz_modu = st.radio(
        "📐 Analiz Modu",
        options=["📅 Günlük", "📊 Güniçi", "🏆 BIST30 Tarama"],
        index=0,
    )

    st.markdown("---")

    if analiz_modu == "🏆 BIST30 Tarama":
        st.markdown("**🏆 En İyi Hisseyi Bul**")
        tarama_baslangic = st.date_input(
            "Başlangıç Tarihi",
            value=date(2025, 1, 1),
            min_value=date(2000, 1, 1),
            max_value=date.today() - pd.Timedelta(days=1),
        )
        n_gun_tarama = tarama_baslangic
        st.markdown(
            "<span style='font-size:0.8em;color:#6b7280'>"
            "Seçilen tarihten bugüne en yüksek getirili hisse seçilir."
            "</span>",
            unsafe_allow_html=True
        )
        ticker_input = "GARAN.IS"
        start_date   = date(1990, 1, 1)
        n_rows       = 60
        intraday_date = None

    elif analiz_modu == "📅 Günlük":
        ticker_input = st.text_input(
            "🔍 Ticker",
            value="GARAN.IS",
            placeholder="Örn: GARAN.IS, AAPL, BTC-USD",
        ).strip().upper()
        st.markdown("---")
        st.markdown("**📅 Başlangıç Tarihi**")
        start_date = st.date_input(
            "Başlangıç",
            value=date(2025, 1, 1),
            min_value=date(1990, 1, 1),
            max_value=date.today(),
            label_visibility="collapsed"
        )
        n_rows = st.slider("Gösterilecek Satır Sayısı", 10, 500, 60, 10)
        intraday_date = None
        n_gun_tarama  = None

    else:  # Güniçi
        ticker_input = st.text_input(
            "🔍 Ticker",
            value="GARAN.IS",
            placeholder="Örn: GARAN.IS, AAPL, BTC-USD",
        ).strip().upper()
        st.markdown("---")
        st.markdown("**📅 Gün Seç (son 60 gün)**")
        intraday_date = st.date_input(
            "Gün",
            value=date.today(),
            min_value=date.today() - pd.Timedelta(days=59),
            max_value=date.today(),
            label_visibility="collapsed"
        )
        start_date    = date(1990, 1, 1)
        n_rows        = 60
        n_gun_tarama  = None

    st.markdown("---")
    secondary_metric = st.radio(
        "📉 Likidite Boyutları",
        options=[
            "Daily Range (%) — Anındalık",
            "Amihud (×10⁶) — Genişlik",
            "Hacim — Derinlik",
            "C-S Spread (%) — Sıkılık",
            "MEC — Esneklik",
        ],
        index=0,
    )
    secondary_metric = secondary_metric.split(" — ")[0]

    with st.expander("📖 Boyut Tanımları"):
        st.markdown("""
**📊 Daily Range — Anındalık**
Günlük yüksek ve düşük fiyat arasındaki mutlak fark.

**📊 Amihud (2002) — Genişlik**
Günlük mutlak getirinin TL hacime oranı (×10⁶).

**📊 Hacim — Derinlik**
Günlük toplam işlem adedi (log₁₀ normalize).

**📊 Corwin-Schultz (2012) — Sıkılık**
Günlük yüksek/düşük fiyat oranından tahmin edilen bid-ask spread.

**📊 MEC — Esneklik**
Haftalık getiri varyansının günlük getiri varyansına oranı (90 günlük rolling).
        """)

    with st.expander("📖 RVOL — Göreceli Hacim"):
        st.markdown("""
**RVOL**, bir zaman dilimindeki işlem hacminin aynı zaman diliminin geçmiş ortalamasına oranıdır.
- `RVOL > 1.5` → Normalin üzerinde hacim
- `RVOL < 0.8` → Zayıf hacim
        """)

    with st.expander("📖 BIST30 Tarama Skoru"):
        st.markdown("""
**Skor Hesabı:**

Seçilen başlangıç tarihinden bugüne BIST30 hisselerinin fiyat getirisi hesaplanır. En yüksek getirili hisse seçilir.
        """)

    st.markdown("---")
    regime_metric = st.radio(
        "🔬 Rejim & Lead-Lag Boyutu",
        options=[
            "Daily Range (%) — Anındalık",
            "Amihud (×10⁶) — Genişlik",
            "Hacim — Derinlik",
            "C-S Spread (%) — Sıkılık",
            "MEC — Esneklik",
        ],
        index=1,
    )
    regime_metric = regime_metric.split(" — ")[0]

    st.markdown("---")
    run = st.button("⚡ Başlat", use_container_width=True, type="primary")
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Otomatik Yenile (55s)", value=False)
    if auto_refresh:
        st.markdown(f"<span style='color:#6b7280;font-size:0.8em'>Son: {datetime.now().strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)
        st.components.v1.html(
            "<script>setTimeout(function(){window.location.reload();}, 55000);</script>",
            height=0
        )

# ── Ana Alan ─────────────────────────────────────────────────────────────────
st.markdown("# 📈 BIST30 Likidite Analizi")

if run or "last_ticker" in st.session_state:
    if run:
        st.session_state["last_ticker"]       = ticker_input
        st.session_state["last_start"]        = str(start_date)
        st.session_state["last_mode"]         = analiz_modu
        st.session_state["last_intraday_date"]= str(intraday_date) if intraday_date else None
        st.session_state["last_n_gun_tarama"] = n_gun_tarama if analiz_modu == "🏆 BIST30 Tarama" else None
        st.session_state["last_regime_metric"] = regime_metric
        st.session_state["tarama_sonuc"]      = None  # her yeni run'da sıfırla

    _ticker        = st.session_state.get("last_ticker", ticker_input)
    _start         = st.session_state.get("last_start", str(start_date))
    _secondary     = secondary_metric
    _mode          = st.session_state.get("last_mode", analiz_modu)
    _intraday_date = st.session_state.get("last_intraday_date")
    _n_gun_tarama  = st.session_state.get("last_n_gun_tarama")
    _regime_metric = st.session_state.get("last_regime_metric", regime_metric)

    # ── BIST30 Tarama Modu ───────────────────────────────────────────────────
    if _mode == "🏆 BIST30 Tarama":
        if run:
            sonuc = fetch_bist30_best(_n_gun_tarama)
            st.session_state["tarama_sonuc"] = sonuc
            if sonuc:
                st.session_state["last_ticker"] = sonuc["ticker"]
                _ticker = sonuc["ticker"]
        else:
            sonuc = st.session_state.get("tarama_sonuc")
            if sonuc:
                _ticker = sonuc["ticker"]

        if sonuc:
            getiri_pct   = sonuc["getiri"] * 100
            ort_hacim_m  = sonuc["ort_hacim"] / 1_000_000
            skor         = sonuc["skor"]
            g_rank_pct   = sonuc["getiri_rank"] * 100

            # Açıklayıcı metin
            _baslangic_str = pd.Timestamp(_n_gun_tarama).strftime("%d.%m.%Y") if hasattr(_n_gun_tarama, 'year') else str(_n_gun_tarama)
            getiri_aciklama = (
                f"BIST30 içinde getiri sıralamasında **%{g_rank_pct:.0f}. persentilde**"
                f" yer aldı — {_baslangic_str} tarihinden bu yana **+%{getiri_pct:.2f}** kazandırdı."
            )
            secim_neden = "BIST30 içinde seçilen dönemde en yüksek getiriyi sağladı."

            st.markdown(f"""
<div class="winner-banner">
<div style="font-family:'IBM Plex Mono',monospace;font-size:1.4em;color:#22c55e;font-weight:700">
🏆 {_ticker}
</div>
<div style="color:#94a3b8;font-size:0.9em;margin-top:6px">
{_baslangic_str} itibarıyla · Getiri: <span style="color:#22c55e;font-weight:600">+%{getiri_pct:.2f}</span> &nbsp;|&nbsp;
Ort. Günlük Hacim: <span style="color:#7dd3fc;font-weight:600">{ort_hacim_m:.0f}M</span>
</div>
<div style="color:#6b7280;font-size:0.82em;margin-top:10px;line-height:1.6">
📈 {getiri_aciklama}<br>
✅ <em>{secim_neden}</em>
</div>
</div>
""", unsafe_allow_html=True)

            # Tüm BIST30 sıralama tablosu (küçük)
            with st.expander("📋 BIST30 Tam Sıralama"):
                df_tum = sonuc["df_tum"].copy()
                df_tum["Getiri (%)"]     = (df_tum["getiri"] * 100).round(2)
                df_tum["Ort. Hacim (M)"] = (df_tum["ort_hacim"] / 1e6).round(1)
                df_tum["Skor"]           = df_tum["skor"].round(3)
                df_tum["Sıra"]           = range(1, len(df_tum) + 1)
                st.dataframe(
                    df_tum[["Sıra", "ticker", "Getiri (%)", "Ort. Hacim (M)", "Skor"]]
                    .rename(columns={"ticker": "Ticker"}),
                    use_container_width=True,
                    hide_index=True,
                )
            if sonuc.get("hatalar"):
                st.caption(f"⚠️ Veri çekilemeyen hisseler: {', '.join(sonuc['hatalar'])}")

            st.markdown("---")
            st.markdown(f"### 📅 {_ticker} — Günlük Likidite Analizi")

        elif run:
            st.error("❌ BIST30 verisi çekilemedi.")
            st.stop()

    # ── 2 Dakikalık Mod ──────────────────────────────────────────────────────
    if _mode == "📊 Güniçi":
        sel_date = _intraday_date or str(date.today())
        with st.spinner(f"{_ticker} güniçi verisi çekiliyor..."):
            df_day  = fetch_intraday(_ticker, sel_date)
            df_60d  = fetch_intraday_60d(_ticker)

        if df_day.empty:
            st.error(f"❌ {_ticker} için {sel_date} tarihinde 2dk veri bulunamadı.")
        else:
            st.markdown(f"### ⏱️ {_ticker} — {pd.Timestamp(sel_date).strftime('%d.%m.%Y')} Güniçi Analiz")
            intra = compute_intraday_metrics(df_day, df_60d)

            open_p  = df_day["Open"].iloc[0]
            close_p = df_day["Close"].iloc[-1]
            high_p  = df_day["High"].max()
            low_p   = df_day["Low"].min()
            gunici_chg = (close_p - open_p) / open_p * 100
            chg_sign = "+" if gunici_chg > 0 else ""

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Açılış",  f"{open_p:.2f}")
            k2.metric("Kapanış", f"{close_p:.2f}", f"{chg_sign}{gunici_chg:.2f}%")
            k3.metric("Yüksek",  f"{high_p:.2f}")
            k4.metric("Düşük",   f"{low_p:.2f}")
            k5.metric("Günlük Range", f"{((high_p - low_p) / low_p * 100):.2f}%")
            st.markdown("---")

            def intraday_yorum(intra, ticker, sel_date):
                if intra.empty or len(intra) < 5:
                    return
                rvol_valid = intra["RVOL"].dropna()
                if len(rvol_valid) >= 3:
                    top3_likit    = rvol_valid.nlargest(3)
                    bottom3_likit = rvol_valid.nsmallest(3)
                times = intra.index
                sabah   = intra[times.hour < 11]
                ogle    = intra[(times.hour >= 11) & (times.hour < 14)]
                kapanis = intra[times.hour >= 14]
                def dilim_rvol(d): return d["RVOL"].mean() if not d.empty and d["RVOL"].notna().any() else np.nan
                sabah_rvol   = dilim_rvol(sabah)
                ogle_rvol    = dilim_rvol(ogle)
                kapanis_rvol = dilim_rvol(kapanis)
                rvol_ort = intra["RVOL"].mean()
                amihud_ort = intra["Amihud (2dk)"].mean()
                cs_valid = intra[intra["C-S Spread (%)"] > 0]["C-S Spread (%)"]
                cs_ort   = cs_valid.mean() if not cs_valid.empty else np.nan
                sinyaller = {}
                if pd.notna(rvol_ort):
                    sinyaller["RVOL"]   = "iyi" if rvol_ort >= 1.2 else ("kötü" if rvol_ort < 0.8 else "nötr")
                if pd.notna(amihud_ort):
                    sinyaller["Amihud"] = "kötü" if amihud_ort > intra["Amihud (2dk)"].quantile(0.75) else "iyi"
                if pd.notna(cs_ort):
                    sinyaller["C-S"]    = "kötü" if cs_ort > cs_valid.quantile(0.75) else "iyi"
                kotu = sum(1 for s in sinyaller.values() if s == "kötü")
                iyi  = sum(1 for s in sinyaller.values() if s == "iyi")
                n    = len(sinyaller)
                if n > 0:
                    if kotu >= n * 0.6:
                        genel, renk, ikon = "Düşük Likidite", "#ef4444", "🔴"
                    elif iyi >= n * 0.6:
                        genel, renk, ikon = "Yüksek Likidite", "#22c55e", "🟢"
                    else:
                        genel, renk, ikon = "Orta Likidite", "#f59e0b", "🟡"
                    st.markdown(f"### {ikon} Güniçi Likidite: <span style='color:{renk}'>{genel}</span>", unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                for col_w, ad, rv in [(d1, "🌅 Sabah (<11:00)", sabah_rvol),
                                       (d2, "☀️ Öğle (11-14)", ogle_rvol),
                                       (d3, "🔔 Kapanış (>14:00)", kapanis_rvol)]:
                    if pd.notna(rv):
                        r = "#22c55e" if rv >= 1.2 else ("#ef4444" if rv < 0.8 else "#f59e0b")
                        col_w.markdown(
                            f"<div style='background:#1e2235;border-left:3px solid {r};padding:10px 12px;border-radius:6px'>"
                            f"<div style='color:#94a3b8;font-size:0.75em'>{ad}</div>"
                            f"<div style='color:{r};font-weight:600'>RVOL: {rv:.2f}</div>"
                            f"</div>", unsafe_allow_html=True)
                st.markdown("")
                paragraf = []
                if pd.notna(rvol_ort):
                    if rvol_ort >= 1.5:
                        paragraf.append(f"Gün genelinde ortalama RVOL **{rvol_ort:.2f}** — normalin belirgin üzerinde hacim var.")
                    elif rvol_ort < 0.8:
                        paragraf.append(f"Gün genelinde ortalama RVOL **{rvol_ort:.2f}** — ince işlem, piyasa ilgisiz.")
                    else:
                        paragraf.append(f"Gün genelinde ortalama RVOL **{rvol_ort:.2f}** — normale yakın hacim.")
                if len(rvol_valid) >= 3:
                    en_yogun  = top3_likit.index.strftime("%H:%M").tolist()
                    en_seyrek = bottom3_likit.index.strftime("%H:%M").tolist()
                    paragraf.append(f"En yoğun saatler: **{', '.join(en_yogun)}**. En seyrek saatler: **{', '.join(en_seyrek)}**.")
                dilimleri = [(sabah_rvol, "sabah"), (ogle_rvol, "öğle"), (kapanis_rvol, "kapanış")]
                gecerli = [(rv, ad) for rv, ad in dilimleri if pd.notna(rv)]
                if gecerli:
                    en_iyi  = max(gecerli, key=lambda x: x[0])
                    en_kotu = min(gecerli, key=lambda x: x[0])
                    if en_iyi[0] != en_kotu[0]:
                        paragraf.append(f"En likit dilim **{en_iyi[1]}** (RVOL: {en_iyi[0]:.2f}), en az likit dilim **{en_kotu[1]}** (RVOL: {en_kotu[0]:.2f}).")
                if pd.notna(cs_ort) and not cs_valid.empty:
                    paragraf.append(f"Ortalama C-S Spread: **%{cs_ort:.4f}** — {'işlem maliyeti yüksek' if sinyaller.get('C-S') == 'kötü' else 'işlem maliyeti normal'}.")
                st.markdown(" ".join(paragraf))

            intraday_yorum(intra, _ticker, sel_date)
            st.markdown("---")

            fig_i = make_subplots(specs=[[{"secondary_y": True}]])
            up_m   = intra["Değişim (%)"] > 0
            down_m = intra["Değişim (%)"] <= 0
            fig_i.add_trace(go.Scatter(x=intra.index, y=intra["Kapanış"], name="Kapanış", line=dict(color="#22c55e", width=1.5)), secondary_y=False)
            fig_i.add_trace(go.Scatter(x=intra.index[up_m], y=intra["Kapanış"][up_m], mode="markers", name="Artış", marker=dict(color="#22c55e", size=4), customdata=intra["Değişim (%)"][up_m], hovertemplate="%{x}<br>%{y}<br>+%{customdata:.3f}%<extra></extra>"), secondary_y=False)
            fig_i.add_trace(go.Scatter(x=intra.index[down_m], y=intra["Kapanış"][down_m], mode="markers", name="Düşüş", marker=dict(color="#ef4444", size=4), customdata=intra["Değişim (%)"][down_m], hovertemplate="%{x}<br>%{y}<br>%{customdata:.3f}%<extra></extra>"), secondary_y=False)
            fig_i.add_trace(go.Bar(x=intra.index, y=intra["Hacim"], name="Hacim", marker_color="#7dd3fc", opacity=0.3), secondary_y=True)
            fig_i.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(family="IBM Plex Mono", color="#94a3b8", size=11), legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=40, b=10), height=380, title=dict(text="Güniçi Fiyat & Hacim", font=dict(color="#94a3b8", size=12)))
            fig_i.update_xaxes(showgrid=False, color="#94a3b8")
            fig_i.update_yaxes(title_text="Kapanış", title_font=dict(color="#22c55e"), tickfont=dict(color="#22c55e"), showgrid=True, gridcolor="#1e2235", secondary_y=False)
            fig_i.update_yaxes(title_text="Hacim", title_font=dict(color="#7dd3fc"), tickfont=dict(color="#7dd3fc"), showgrid=False, secondary_y=True)
            st.plotly_chart(fig_i, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            fig_l = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["RVOL", "Bar Range (%) & C-S Spread (%)", "Amihud (2dk)"], vertical_spacing=0.08)
            rvol_colors = ["#22c55e" if v >= 1.2 else ("#ef4444" if v < 0.8 else "#f59e0b") for v in intra["RVOL"].fillna(1.0)]
            fig_l.add_trace(go.Bar(x=intra.index, y=intra["RVOL"], name="RVOL", marker_color=rvol_colors, opacity=0.8), row=1, col=1)
            fig_l.add_hline(y=1.0, line=dict(color="#6b7280", dash="dot", width=1), row=1, col=1)
            fig_l.add_trace(go.Scatter(x=intra.index, y=intra["Bar Range (%)"], name="Bar Range (%)", line=dict(color="#7dd3fc", width=1.2)), row=2, col=1)
            cs_plot = intra["C-S Spread (%)"].replace(0, np.nan)
            fig_l.add_trace(go.Scatter(x=intra.index, y=cs_plot, name="C-S Spread (%)", line=dict(color="#a78bfa", width=1.2)), row=2, col=1)
            amihud_log = intra["Amihud (2dk)"].apply(lambda x: abs(np.log10(x)) if pd.notna(x) and x > 0 else np.nan)
            fig_l.add_trace(go.Scatter(x=intra.index, y=amihud_log, name="log|Amihud (2dk)|", line=dict(color="#f59e0b", width=1.2)), row=3, col=1)
            fig_l.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(family="IBM Plex Mono", color="#94a3b8", size=11), legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=50, b=10), height=500, showlegend=True)
            for i in range(1, 4):
                fig_l.update_xaxes(showgrid=False, color="#94a3b8", row=i, col=1)
                fig_l.update_yaxes(showgrid=True, gridcolor="#1e2235", color="#94a3b8", row=i, col=1)
            st.plotly_chart(fig_l, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
            st.markdown("---")

            cols_intra = ["Kapanış", "Açılış", "Yüksek", "Düşük", "Hacim", "Değişim (%)", "Bar Range (%)", "RVOL", "Amihud (2dk)", "C-S Spread (%)"]
            disp_intra = intra[cols_intra].iloc[::-1]
            header_i = "<tr><th>Zaman</th>" + "".join(f"<th>{c}</th>" for c in cols_intra) + "</tr>"
            rows_i = ""
            for idx, row in disp_intra.iterrows():
                zaman = idx.strftime("%H:%M")
                def cv(val, col):
                    if pd.isna(val): return '<span class="neutral">—</span>'
                    if col == "Değişim (%)":
                        cls = "pos" if val > 0 else ("neg" if val < 0 else "neutral")
                        sign = "+" if val > 0 else ""
                        return f'<span class="{cls}">{sign}{val:.3f}%</span>'
                    if col == "RVOL":
                        cls = "pos" if val >= 1.2 else ("neg" if val < 0.8 else "neutral")
                        return f'<span class="{cls}">{val:.3f}</span>'
                    if col == "Hacim":
                        return f'<span class="neutral">{int(val):,}</span>'
                    if col == "Amihud (2dk)":
                        lv = abs(np.log10(val)) if val > 0 else np.nan
                        return f'<span class="neutral">{lv:.2f}</span>' if pd.notna(lv) else '<span class="neutral">—</span>'
                    return f'<span class="neutral">{val:.4f}</span>'
                cells = "".join(f"<td>{cv(row[c], c)}</td>" for c in cols_intra)
                rows_i += f"<tr><td><span style='font-family:IBM Plex Mono;font-size:0.85em;color:#94a3b8'>{zaman}</span></td>{cells}</tr>"
            tbl_html = f"""<style>
            body{{margin:0;background:#0f1117;}}
            .pos{{color:#22c55e;font-weight:600;}}
            .neg{{color:#ef4444;font-weight:600;}}
            .neutral{{color:#94a3b8;}}
            .data-table{{width:100%;border-collapse:collapse;font-size:0.82em;margin-top:8px;background:#0f1117;}}
            .data-table th{{background:#1e2235;color:#7dd3fc;font-family:'IBM Plex Mono',monospace;font-weight:600;padding:10px 12px;text-align:right;border-bottom:2px solid #2a2d3e;white-space:nowrap;}}
            .data-table th:first-child{{text-align:left;}}
            .data-table td{{padding:8px 12px;text-align:right;border-bottom:1px solid #1e2235;background:#0f1117;color:#94a3b8;}}
            .data-table td:first-child{{text-align:left;}}
            .data-table tr:hover td{{background:#141824;}}</style>
            <div style="overflow-x:auto;max-height:60vh;overflow-y:auto;background:#0f1117;">
            <table class="data-table"><thead>{header_i}</thead><tbody>{rows_i}</tbody></table></div>"""
            st.components.v1.html(tbl_html, height=600, scrolling=True)

    # ── Günlük Mod (Günlük + BIST30 tarama sonrası) ──────────────────────────
    if _mode in ("📅 Günlük", "🏆 BIST30 Tarama"):
        with st.spinner(f"{_ticker} verisi çekiliyor..."):
            raw  = fetch_data(_ticker, _start)
            live = fetch_live(_ticker)

        if live is not None and not raw.empty:
            today_ts = pd.Timestamp(date.today())
            if today_ts not in raw.index:
                raw = pd.concat([raw, live.to_frame().T])
            else:
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    raw.at[today_ts, col] = live[col]

        if raw.empty:
            st.error(f"❌ {_ticker} için veri bulunamadı.")
        else:
            oldest = fetch_oldest_date(_ticker)
            newest = raw.index.max().strftime("%d.%m.%Y")
            total  = len(raw)

            col1, col2, col3, col4 = st.columns(4)
            last_close = raw["Close"].iloc[-1]
            prev_close = raw["Close"].iloc[-2] if len(raw) > 1 else last_close
            chg = ((last_close - prev_close) / prev_close) * 100
            chg_sign = "+" if chg > 0 else ""
            col1.metric("Ticker", f"{_ticker}")
            col2.metric("Son Kapanış", f"{last_close:.2f}", f"{chg_sign}{chg:.2f}%")
            col3.metric("En Eski Veri", oldest)
            col4.metric("Toplam Gün", f"{total:,}")
            st.markdown("---")

            metrics = compute_metrics(raw)
            display = metrics.iloc[::-1].head(n_rows)

            up_days   = metrics[metrics["Güniçi Değ. (%)"] > 0]
            down_days = metrics[metrics["Güniçi Değ. (%)"] < 0]
            avg_range_up_tl    = up_days["Daily Range (₺)"].mean()
            avg_range_down_tl  = down_days["Daily Range (₺)"].mean()
            avg_range_up_pct   = up_days["Daily Range (%)"].mean()
            avg_range_down_pct = down_days["Daily Range (%)"].mean()

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("📗 Artış Günü Ort. Range (₺)", f"{avg_range_up_tl:.2f}", f"{avg_range_up_pct:.2f}%")
            sc2.metric("📗 Artış Günü Sayısı", f"{len(up_days):,}")
            sc3.metric("📕 Düşüş Günü Ort. Range (₺)", f"{avg_range_down_tl:.2f}", f"{avg_range_down_pct:.2f}%", delta_color="off")
            sc4.metric("📕 Düşüş Günü Sayısı", f"{len(down_days):,}")
            st.markdown("---")

            def likidite_yorum(metrics):
                m = metrics.dropna(subset=["Daily Range (%)", "Amihud (×10⁶)", "Hacim", "C-S Spread (%)", "MEC"])
                if len(m) < 21:
                    st.info("Yorum için yeterli veri yok (min. 21 gün).")
                    return
                son     = m.iloc[-1]
                trend_w = 20
                def pct(series, val): return round((series < val).mean() * 100, 1)
                dr_pct    = pct(m["Daily Range (%)"], son["Daily Range (%)"])
                amihud_v  = abs(np.log10(son["Amihud (×10⁶)"])) if son["Amihud (×10⁶)"] > 0 else np.nan
                amihud_s  = m["Amihud (×10⁶)"].apply(lambda x: abs(np.log10(x)) if x > 0 else np.nan).dropna()
                amihud_pct= pct(amihud_s, amihud_v) if amihud_v else 50
                hacim_pct = pct(m["Hacim"], son["Hacim"])
                cs_valid  = m[m["C-S Spread (%)"] > 0]["C-S Spread (%)"]
                cs_son    = cs_valid.iloc[-1] if len(cs_valid) > 0 else None
                cs_pct    = pct(cs_valid, cs_son) if cs_son is not None else None
                mec_pct   = pct(m["MEC"].dropna(), son["MEC"]) if pd.notna(son["MEC"]) else None
                def trend(series):
                    s = series.dropna()
                    if len(s) < trend_w * 2: return 0
                    return s.iloc[-trend_w:].mean() - s.iloc[-trend_w*2:-trend_w].mean()
                dr_trend     = trend(m["Daily Range (%)"])
                amihud_trend = trend(amihud_s)
                hacim_trend  = trend(m["Hacim"])
                cs_trend     = trend(cs_valid) if len(cs_valid) >= trend_w * 2 else 0
                mec_trend    = trend(m["MEC"].dropna())
                sinyaller = {}
                def sinyal_ters(pct_val, tr):
                    s = "kötü" if pct_val >= 75 else ("nötr" if pct_val >= 50 else "iyi")
                    t = "↑" if tr > 0 else ("↓" if tr < 0 else "→")
                    return s, t
                def sinyal_duz(pct_val, tr):
                    s = "iyi" if pct_val >= 75 else ("nötr" if pct_val >= 50 else "kötü")
                    t = "↑" if tr > 0 else ("↓" if tr < 0 else "→")
                    return s, t
                sinyaller["Daily Range"]  = sinyal_ters(dr_pct,     dr_trend)
                sinyaller["Amihud"]       = sinyal_ters(amihud_pct, amihud_trend)
                sinyaller["Hacim"]        = sinyal_duz (hacim_pct,  hacim_trend)
                if cs_pct is not None:
                    sinyaller["C-S Spread"] = sinyal_ters(cs_pct, cs_trend)
                if mec_pct is not None:
                    mec_sinyal  = "kötü" if son["MEC"] > 1 else ("nötr" if son["MEC"] > 0.8 else "iyi")
                    mec_trend_ok= "↑" if mec_trend > 0 else ("↓" if mec_trend < 0 else "→")
                    sinyaller["MEC"] = (mec_sinyal, mec_trend_ok)
                kotu   = sum(1 for s, _ in sinyaller.values() if s == "kötü")
                iyi    = sum(1 for s, _ in sinyaller.values() if s == "iyi")
                toplam = len(sinyaller)
                if kotu >= toplam * 0.6:   genel, renk, ikon = "Düşük Likidite",  "#ef4444", "🔴"
                elif iyi >= toplam * 0.6:  genel, renk, ikon = "Yüksek Likidite", "#22c55e", "🟢"
                else:                      genel, renk, ikon = "Orta Likidite",    "#f59e0b", "🟡"
                renk_map   = {"iyi": "#22c55e", "nötr": "#94a3b8", "kötü": "#ef4444"}
                etiket_map = {
                    "Daily Range":  ("Anındalık",  f"%{dr_pct:.0f} persentil"),
                    "Amihud":       ("Genişlik",   f"%{amihud_pct:.0f} persentil"),
                    "Hacim":        ("Derinlik",   f"%{hacim_pct:.0f} persentil"),
                    "C-S Spread":   ("Sıkılık",    f"%{cs_pct:.0f} persentil" if cs_pct is not None else "—"),
                    "MEC":          ("Esneklik",   f"MEC = {son['MEC']:.3f}" if pd.notna(son["MEC"]) else "—"),
                }
                st.markdown(f"### {ikon} Likidite Durumu: <span style='color:{renk}'>{genel}</span>", unsafe_allow_html=True)
                boyut_cols = st.columns(len(sinyaller))
                for col_i, (boyut, (sinyal, trend_ok)) in enumerate(sinyaller.items()):
                    r = renk_map[sinyal]
                    ad, detay = etiket_map[boyut]
                    boyut_cols[col_i].markdown(
                        f"<div style='background:#1e2235;border-left:3px solid {r};padding:10px 12px;border-radius:6px'>"
                        f"<div style='color:#94a3b8;font-size:0.75em;font-family:IBM Plex Mono'>{boyut}</div>"
                        f"<div style='color:{r};font-weight:600;font-size:1em'>{ad}</div>"
                        f"<div style='color:#94a3b8;font-size:0.78em'>{detay} {trend_ok}</div>"
                        f"</div>", unsafe_allow_html=True)
                st.markdown("")
                paragraf = []
                if genel == "Düşük Likidite":    paragraf.append(f"**{_ticker}** bugün itibarıyla **düşük likidite** koşullarında işlem görüyor.")
                elif genel == "Yüksek Likidite": paragraf.append(f"**{_ticker}** bugün itibarıyla **yüksek likidite** koşullarında işlem görüyor.")
                else:                             paragraf.append(f"**{_ticker}** bugün itibarıyla **karma likidite** koşullarında işlem görüyor.")
                if sinyaller.get("Amihud", ("nötr",))[0] == "kötü" and sinyaller.get("Hacim", ("nötr",))[0] == "kötü":
                    paragraf.append("Fiyat etkisi yüksek, işlem hacmi düşük — büyük emirler ciddi kayma yaratabilir.")
                elif sinyaller.get("Amihud", ("nötr",))[0] == "iyi" and sinyaller.get("Hacim", ("nötr",))[0] == "iyi":
                    paragraf.append("Fiyat etkisi düşük ve hacim güçlü — emir gerçekleştirme maliyeti tarihsel olarak düşük.")
                elif sinyaller.get("Amihud", ("nötr",))[0] != sinyaller.get("Hacim", ("nötr",))[0]:
                    paragraf.append("Amihud ve hacim sinyalleri çelişiyor — likidite konusunda temkinli olmak gerekir.")
                if sinyaller.get("Daily Range", ("nötr",))[0] == "kötü":
                    paragraf.append(f"Günlük fiyat aralığı tarihsel dağılımın %{dr_pct:.0f}'lik dilimine girmiş; anındalık zayıf.")
                if cs_pct is not None and sinyaller.get("C-S Spread", ("nötr",))[0] == "kötü":
                    paragraf.append(f"Bid-ask spread (son geçerli: %{cs_son:.4f}) %{cs_pct:.0f} persentilde — işlem maliyeti yüksek.")
                if pd.notna(son.get("MEC")):
                    if son["MEC"] > 1:   paragraf.append(f"MEC = {son['MEC']:.3f} (>1): Fiyat yeni dengesine yavaş dönüyor, piyasa esnekliği zayıf.")
                    else:                paragraf.append(f"MEC = {son['MEC']:.3f} (≤1): Fiyat yeni dengesine hızlı dönüyor, piyasa dayanıklı.")
                kotu_trend = sum(1 for _, t in sinyaller.values() if t == "↑" and _ == "kötü")
                iyi_trend  = sum(1 for _, t in sinyaller.values() if t == "↓" and _ == "kötü")
                if kotu_trend >= 2: paragraf.append("Son 20 günlük trend likiditenin **kötüleştiğine** işaret ediyor.")
                elif iyi_trend >= 2: paragraf.append("Son 20 günlük trend likiditenin **iyileştiğine** işaret ediyor.")
                st.markdown(" ".join(paragraf))

            likidite_yorum(metrics)
            st.markdown("---")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics["Kapanış (₺)"], name="Kapanış", line=dict(color="#22c55e", width=1.5)), secondary_y=False)
            up_mask   = metrics["Günlük Değ. (%)"] > 0
            down_mask = metrics["Günlük Değ. (%)"] < 0
            fig.add_trace(go.Scatter(x=metrics.index[up_mask], y=metrics["Kapanış (₺)"][up_mask], mode="markers", name="Artış Günü", marker=dict(color="#22c55e", size=4, symbol="circle"), hovertemplate="%{x}<br>Kapanış: %{y}<br>Değ: +%{customdata:.2f}%<extra></extra>", customdata=metrics["Günlük Değ. (%)"][up_mask]), secondary_y=False)
            fig.add_trace(go.Scatter(x=metrics.index[down_mask], y=metrics["Kapanış (₺)"][down_mask], mode="markers", name="Düşüş Günü", marker=dict(color="#ef4444", size=4, symbol="circle"), hovertemplate="%{x}<br>Kapanış: %{y}<br>Değ: %{customdata:.2f}%<extra></extra>", customdata=metrics["Günlük Değ. (%)"][down_mask]), secondary_y=False)
            sec_col  = _secondary
            sec_data = metrics[sec_col].dropna()
            if sec_col == "Amihud (×10⁶)":
                log_amihud = np.log10(sec_data.replace(0, np.nan).dropna()).abs()
                fig.add_trace(go.Scatter(x=log_amihud.index, y=log_amihud.values, name="log₁₀(Amihud)", line=dict(color="#f59e0b", width=1.2)), secondary_y=True)
            elif sec_col == "Hacim":
                log_hacim = np.log10(metrics["Hacim"].replace(0, np.nan).dropna())
                fig.add_trace(go.Scatter(x=log_hacim.index, y=log_hacim.values, name="log₁₀(Hacim)", line=dict(color="#7dd3fc", width=1.2)), secondary_y=True)
            elif sec_col == "C-S Spread (%)":
                fig.add_trace(go.Scatter(x=sec_data.index, y=sec_data.values, name="C-S Spread (%)", line=dict(color="#a78bfa", width=1.2)), secondary_y=True)
            elif sec_col == "MEC":
                fig.add_trace(go.Scatter(x=sec_data.index, y=sec_data.values, name="MEC", line=dict(color="#fb923c", width=1.2)), secondary_y=True)
                fig.add_hline(y=1.0, line=dict(color="#6b7280", dash="dot", width=1), secondary_y=True)
            else:
                window = min(30, len(sec_data))
                trend_vals = []
                for i in range(len(sec_data)):
                    start_i = max(0, i - window + 1)
                    segment = sec_data.iloc[start_i:i+1]
                    x_seg   = np.arange(len(segment))
                    if len(segment) >= 2:
                        z = np.polyfit(x_seg, segment.values, 1)
                        trend_vals.append(np.poly1d(z)(len(segment) - 1))
                    else:
                        trend_vals.append(segment.iloc[-1])
                fig.add_trace(go.Scatter(x=sec_data.index, y=trend_vals, name=f"{sec_col} Trend", line=dict(color="#f59e0b", width=1.8)), secondary_y=True)

            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(family="IBM Plex Mono", color="#94a3b8", size=11), legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)), margin=dict(l=10, r=10, t=40, b=10), height=400, dragmode="pan")
            fig.update_xaxes(showgrid=False, color="#94a3b8", rangeslider=dict(visible=True, bgcolor="#1e2235", thickness=0.06), rangeselector=dict(bgcolor="#1e2235", activecolor="#7dd3fc", buttons=list([dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"), dict(count=3, label="3Y", step="year", stepmode="backward"), dict(step="all", label="Tümü")])))
            fig.update_yaxes(title_text="Kapanış", title_font=dict(color="#22c55e"), tickfont=dict(color="#22c55e"), showgrid=True, gridcolor="#1e2235", secondary_y=False)
            fig.update_yaxes(title_text=("log₁₀(Amihud)" if sec_col=="Amihud (×10⁶)" else "log₁₀(Hacim)" if sec_col=="Hacim" else "C-S Spread (%)" if sec_col=="C-S Spread (%)" else "MEC" if sec_col=="MEC" else sec_col), title_font=dict(color="#7dd3fc"), tickfont=dict(color="#7dd3fc"), showgrid=False, secondary_y=True, type="linear", range=[0, 6] if sec_col=="Daily Range (%)" else None)
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "modeBarButtonsToAdd": ["pan2d"], "displayModeBar": True})
            st.markdown("---")

            cols_show = ["Kapanış (₺)", "Açılış (₺)", "Yüksek (₺)", "Düşük (₺)", "Hacim", "Günlük Değ. (%)", "Güniçi Değ. (%)", "Daily Range (₺)", "Daily Range (%)", "Amihud (×10⁶)", "log₁₀(Hacim)", "C-S Spread (%)", "MEC"]
            st.markdown("<span style='font-size:0.78em;color:#6b7280;font-family:IBM Plex Mono'>ℹ️ Amihud sütunu: <b>|log₁₀(Amihud)|</b> — yüksek = daha az likit, düşük = daha likit</span>", unsafe_allow_html=True)
            header = "<tr><th>Tarih</th>" + "".join(f"<th>{c}</th>" for c in cols_show) + "</tr>"
            rows = ""
            for idx, row in display.iterrows():
                date_str = idx.strftime("%d.%m.%Y")
                cells = "".join(f"<td>{color_val(row[c], c)}</td>" for c in cols_show)
                rows += f"<tr><td><span style='font-family:IBM Plex Mono;font-size:0.85em;color:#94a3b8'>{date_str}</span></td>{cells}</tr>"
            table_html = f"""<style>
            body{{margin:0;background:#0f1117;}}
            .pos{{color:#22c55e;font-weight:600;}}
            .neg{{color:#ef4444;font-weight:600;}}
            .neutral{{color:#94a3b8;}}
            .data-table{{width:100%;border-collapse:collapse;font-size:0.82em;margin-top:8px;background:#0f1117;}}
            .data-table th{{background:#1e2235;color:#7dd3fc;font-family:'IBM Plex Mono',monospace;font-weight:600;padding:10px 12px;text-align:right;border-bottom:2px solid #2a2d3e;white-space:nowrap;}}
            .data-table th:first-child{{text-align:left;}}
            .data-table td{{padding:8px 12px;text-align:right;border-bottom:1px solid #1e2235;background:#0f1117;color:#94a3b8;}}
            .data-table td:first-child{{text-align:left;}}
            .data-table tr:hover td{{background:#141824;}}</style>
            <div style="overflow-x:auto;max-height:65vh;overflow-y:auto;background:#0f1117;">
            <table class="data-table"><thead>{header}</thead><tbody>{rows}</tbody></table></div>"""
            st.components.v1.html(table_html, height=700, scrolling=True)

            st.markdown("---")
            st.markdown("### 🔗 Likidite Boyutları İlişki Analizi")
            ana = pd.DataFrame({
                "Close":        metrics["Kapanış (₺)"],
                "Daily Range":  metrics["Daily Range (%)"],
                "Amihud (log)": metrics["Amihud (×10⁶)"].apply(lambda x: abs(np.log10(x)) if pd.notna(x) and x > 0 else np.nan),
                "Hacim (log)":  metrics["log₁₀(Hacim)"],
                "C-S Spread":   metrics["C-S Spread (%)"],
                "MEC":          metrics["MEC"],
            }).dropna()
            from scipy.stats import spearmanr
            cols3 = ["Close", "Daily Range", "Amihud (log)", "Hacim (log)", "C-S Spread", "MEC"]
            n = len(cols3)
            corr_matrix = np.zeros((n, n))
            for i, c1 in enumerate(cols3):
                for j, c2 in enumerate(cols3):
                    r, _ = spearmanr(ana[c1], ana[c2])
                    corr_matrix[i][j] = round(r, 3)
            heat_fig = go.Figure(go.Heatmap(z=corr_matrix, x=cols3, y=cols3, colorscale=[[0,"#ef4444"],[0.5,"#1e2235"],[1,"#22c55e"]], zmin=-1, zmax=1, text=corr_matrix.round(2), texttemplate="%{text}", textfont=dict(size=12, family="IBM Plex Mono"), showscale=True))
            heat_fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(family="IBM Plex Mono", color="#94a3b8", size=11), margin=dict(l=10, r=10, t=30, b=10), height=380, title=dict(text="Spearman Korelasyon Matrisi", font=dict(color="#94a3b8", size=12)))
            st.plotly_chart(heat_fig, use_container_width=True)

            st.markdown("**Rolling Spearman Korelasyon (60 gün)**")
            roll_window = 60
            pairs = [("Close","Daily Range","#7dd3fc"),("Close","Amihud (log)","#f59e0b"),("Close","Hacim (log)","#22c55e"),("Close","C-S Spread","#a78bfa"),("Close","MEC","#fb923c")]
            roll_fig = go.Figure()
            for c1, c2, color in pairs:
                roll_corr = [spearmanr(ana[c1].iloc[max(0,i-roll_window):i+1], ana[c2].iloc[max(0,i-roll_window):i+1])[0] if i >= 10 else np.nan for i in range(len(ana))]
                roll_fig.add_trace(go.Scatter(x=ana.index, y=roll_corr, name=f"{c1} × {c2}", line=dict(color=color, width=1.5)))
            roll_fig.add_hline(y=0, line=dict(color="#4b5563", dash="dot", width=1))
            roll_fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(family="IBM Plex Mono", color="#94a3b8", size=11), legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=30, b=10), height=300, yaxis=dict(range=[-1,1], showgrid=True, gridcolor="#1e2235"), xaxis=dict(showgrid=False))
            st.plotly_chart(roll_fig, use_container_width=True, config={"scrollZoom": True, "dragmode": "pan"})

            st.markdown("**Volatilite Rejimi (Daily Range medyanı bazlı)**")
            median_dr = ana["Daily Range"].median()
            ana["Rejim"] = ana["Daily Range"].apply(lambda x: "Yüksek Vol." if x >= median_dr else "Düşük Vol.")
            reg_fig = go.Figure()
            for rejim, color, _ in [("Yüksek Vol.","#ef4444","solid"),("Düşük Vol.","#22c55e","solid")]:
                mask = ana["Rejim"] == rejim
                reg_fig.add_trace(go.Scatter(x=ana.index[mask], y=ana["Close"][mask], mode="markers", name=rejim, marker=dict(color=color, size=3, opacity=0.6)))
            reg_fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(family="IBM Plex Mono", color="#94a3b8", size=11), legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=30, b=10), height=300, yaxis=dict(title="Kapanış", showgrid=True, gridcolor="#1e2235"), xaxis=dict(showgrid=False))
            st.plotly_chart(reg_fig, use_container_width=True, config={"scrollZoom": True, "dragmode": "pan"})

            st.markdown("---")
            import io

            # ── Rejim Tespiti & Lead-Lag Analizi ─────────────────────────────
            st.markdown("### 🔬 Rejim Tespiti & Lead-Lag Analizi")

            # Seçilen boyutu hazırla
            rm = _regime_metric
            if rm == "Amihud (×10⁶)":
                regime_series = metrics["Amihud (×10⁶)"].apply(
                    lambda x: abs(np.log10(x)) if pd.notna(x) and x > 0 else np.nan)
                rm_label = "log|Amihud|"
            elif rm == "Hacim":
                regime_series = metrics["log₁₀(Hacim)"]
                rm_label = "log₁₀(Hacim)"
            elif rm == "Daily Range (%)":
                regime_series = metrics["Daily Range (%)"]
                rm_label = "Daily Range (%)"
            elif rm == "C-S Spread (%)":
                regime_series = metrics["C-S Spread (%)"].replace(0, np.nan)
                rm_label = "C-S Spread (%)"
            else:  # MEC
                regime_series = metrics["MEC"]
                rm_label = "MEC"

            regime_series = regime_series.dropna()
            close_aligned = metrics["Kapanış (₺)"].reindex(regime_series.index)

            # 3 rejim: percentile bazlı
            p33 = regime_series.quantile(0.33)
            p67 = regime_series.quantile(0.67)

            # Amihud, C-S Spread, Daily Range, MEC için: yüksek = kötü likidite
            # Hacim için: düşük = kötü likidite
            if rm == "Hacim":
                def rejim_ata(v):
                    if v >= p67: return "Likit"
                    elif v >= p33: return "Normal"
                    else: return "İlikit"
            else:
                def rejim_ata(v):
                    if v <= p33: return "Likit"
                    elif v <= p67: return "Normal"
                    else: return "İlikit"

            rejim_ser = regime_series.apply(rejim_ata)

            # ── Rejim Grafiği: fiyat + renkli arka plan ──────────────────────
            st.markdown(f"**Likidite Rejimi — {rm_label} bazlı (3 seviye)**")

            rejim_fig = go.Figure()

            # Arka plan renk bantları
            renk_rejim = {"Likit": "rgba(34,197,94,0.10)", "Normal": "rgba(245,158,11,0.10)", "İlikit": "rgba(239,68,68,0.10)"}
            dates_list = regime_series.index.tolist()
            prev_rejim = None
            band_start = None
            for i, (dt, rv) in enumerate(rejim_ser.items()):
                if rv != prev_rejim:
                    if prev_rejim is not None:
                        rejim_fig.add_vrect(
                            x0=band_start, x1=dt,
                            fillcolor=renk_rejim[prev_rejim],
                            layer="below", line_width=0,
                        )
                    band_start = dt
                    prev_rejim = rv
            if prev_rejim is not None:
                rejim_fig.add_vrect(
                    x0=band_start, x1=dates_list[-1],
                    fillcolor=renk_rejim[prev_rejim],
                    layer="below", line_width=0,
                )

            # Fiyat çizgisi
            rejim_fig.add_trace(go.Scatter(
                x=close_aligned.index, y=close_aligned.values,
                name="Kapanış", line=dict(color="#22c55e", width=1.5),
            ))

            # Rejim serisi (ikincil eksen)
            rejim_fig.add_trace(go.Scatter(
                x=regime_series.index, y=regime_series.values,
                name=rm_label, line=dict(color="#7dd3fc", width=1, dash="dot"),
                yaxis="y2", opacity=0.6,
            ))

            rejim_fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
                legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=40, b=10), height=380,
                yaxis=dict(title="Kapanış", title_font=dict(color="#22c55e"),
                           tickfont=dict(color="#22c55e"), showgrid=True, gridcolor="#1e2235"),
                yaxis2=dict(title=rm_label, title_font=dict(color="#7dd3fc"),
                            tickfont=dict(color="#7dd3fc"), overlaying="y", side="right", showgrid=False),
                xaxis=dict(showgrid=False, color="#94a3b8"),
            )

            # Legend açıklaması
            for rejim_ad, renk in [("🟢 Likit","#22c55e"),("🟡 Normal","#f59e0b"),("🔴 İlikit","#ef4444")]:
                rejim_fig.add_annotation(
                    text=rejim_ad, xref="paper", yref="paper",
                    x=0.01 + list(renk_rejim.keys()).index(rejim_ad.split(" ")[1]) * 0.12,
                    y=-0.08, showarrow=False,
                    font=dict(color=renk, size=11, family="IBM Plex Mono"),
                )

            st.plotly_chart(rejim_fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            # ── Rejim İstatistikleri ──────────────────────────────────────────
            st.markdown("**Rejim Bazlı Fiyat Performansı**")
            rej_stats = []
            gunluk_degisim = metrics["Günlük Değ. (%)"].reindex(regime_series.index)
            for r_ad in ["Likit", "Normal", "İlikit"]:
                mask_r = rejim_ser == r_ad
                gun_sayisi = mask_r.sum()
                if gun_sayisi == 0:
                    continue
                degisimler = gunluk_degisim[mask_r].dropna()
                rej_stats.append({
                    "Rejim": r_ad,
                    "Gün Sayısı": int(gun_sayisi),
                    "Ort. Günlük Değ. (%)": round(degisimler.mean(), 3),
                    "Std. Sapma (%)": round(degisimler.std(), 3),
                    "Pozitif Gün (%)": round((degisimler > 0).mean() * 100, 1),
                    "Maks. Getiri (%)": round(degisimler.max(), 2),
                    "Min. Getiri (%)": round(degisimler.min(), 2),
                })

            rej_df = pd.DataFrame(rej_stats)
            renk_map_r = {"Likit": "#22c55e", "Normal": "#f59e0b", "İlikit": "#ef4444"}
            rej_header = "<tr>" + "".join(f"<th>{c}</th>" for c in rej_df.columns) + "</tr>"
            rej_rows = ""
            for _, row in rej_df.iterrows():
                r_renk = renk_map_r.get(row["Rejim"], "#94a3b8")
                cells = f"<td><span style='color:{r_renk};font-weight:600'>{row['Rejim']}</span></td>"
                cells += f"<td><span style='color:#94a3b8'>{int(row['Gün Sayısı'])}</span></td>"
                ort = row["Ort. Günlük Değ. (%)"]
                ort_renk = "#22c55e" if ort > 0 else "#ef4444"
                cells += f"<td><span style='color:{ort_renk}'>{ort:+.3f}%</span></td>"
                cells += f"<td><span style='color:#94a3b8'>{row['Std. Sapma (%)']:.3f}%</span></td>"
                cells += f"<td><span style='color:#94a3b8'>{row['Pozitif Gün (%)']:.1f}%</span></td>"
                cells += f"<td><span style='color:#22c55e'>{row['Maks. Getiri (%)']:+.2f}%</span></td>"
                cells += f"<td><span style='color:#ef4444'>{row['Min. Getiri (%)']:+.2f}%</span></td>"
                rej_rows += f"<tr>{cells}</tr>"

            rej_tbl = f"""<style>
            body{{margin:0;background:#0f1117;}}
            .rt{{width:100%;border-collapse:collapse;font-size:0.82em;background:#0f1117;}}
            .rt th{{background:#1e2235;color:#7dd3fc;font-family:'IBM Plex Mono',monospace;font-weight:600;padding:10px 12px;text-align:right;border-bottom:2px solid #2a2d3e;white-space:nowrap;}}
            .rt th:first-child{{text-align:left;}}
            .rt td{{padding:8px 12px;text-align:right;border-bottom:1px solid #1e2235;background:#0f1117;}}
            .rt td:first-child{{text-align:left;}}
            .rt tr:hover td{{background:#141824;}}</style>
            <div style="background:#0f1117;">
            <table class="rt"><thead><tr>{rej_header}</tr></thead><tbody>{rej_rows}</tbody></table></div>"""
            st.components.v1.html(rej_tbl, height=160, scrolling=False)

            # ── Lead-Lag Analizi ──────────────────────────────────────────────
            st.markdown(f"**Lead-Lag Analizi — {rm_label} → Fiyat Değişimi**")
            st.markdown(
                "<span style='font-size:0.8em;color:#6b7280'>"
                "Her lag için: bugünkü likidite boyutu ile X gün sonraki fiyat değişimi arasındaki Spearman korelasyonu. "
                "Koyu renkli barlar istatistiksel olarak anlamlı (p &lt; 0.05).</span>",
                unsafe_allow_html=True
            )

            from scipy.stats import spearmanr as sp_corr
            lags = [1, 2, 3, 5, 7, 10, 15, 20]
            lag_corrs = []
            lag_pvals = []
            fwd_return = metrics["Günlük Değ. (%)"].reindex(regime_series.index)

            for lag in lags:
                future_ret = fwd_return.shift(-lag)
                combined = pd.DataFrame({"liq": regime_series, "ret": future_ret}).dropna()
                if len(combined) > 20:
                    r, p = sp_corr(combined["liq"], combined["ret"])
                    lag_corrs.append(round(r, 4))
                    lag_pvals.append(round(p, 4))
                else:
                    lag_corrs.append(np.nan)
                    lag_pvals.append(np.nan)

            bar_colors = []
            for r, p in zip(lag_corrs, lag_pvals):
                if pd.isna(r): bar_colors.append("#4b5563")
                elif p < 0.05:
                    bar_colors.append("#22c55e" if r < 0 else "#ef4444")
                else:
                    bar_colors.append("rgba(34,197,94,0.3)" if r < 0 else "rgba(239,68,68,0.3)")

            ll_fig = go.Figure()
            ll_fig.add_bar(
                x=[f"+{l}g" for l in lags],
                y=lag_corrs,
                marker_color=bar_colors,
                text=[f"{r:.3f}" if not pd.isna(r) else "—" for r in lag_corrs],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10, color="#94a3b8"),
                hovertemplate="Lag: %{x}<br>Korelasyon: %{y:.4f}<br>p=%{customdata:.4f}<extra></extra>",
                customdata=lag_pvals,
            )
            ll_fig.add_hline(y=0, line=dict(color="#4b5563", width=1))
            ll_fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
                margin=dict(l=10, r=10, t=30, b=40), height=320,
                xaxis=dict(title="Gecikme (gün)", showgrid=False, color="#94a3b8"),
                yaxis=dict(title="Spearman r", showgrid=True, gridcolor="#1e2235",
                           zeroline=False, range=[-1, 1]),
            )
            ll_fig.add_annotation(
                text="🟢/🔴 koyu = p<0.05 (anlamlı) · soluk = p≥0.05 (anlamsız)",
                xref="paper", yref="paper", x=0.5, y=-0.18,
                showarrow=False, font=dict(size=10, color="#6b7280", family="IBM Plex Mono"),
            )
            st.plotly_chart(ll_fig, use_container_width=True)

            # Özet yorum
            anlamli = [(lags[i], lag_corrs[i]) for i in range(len(lags))
                       if not pd.isna(lag_corrs[i]) and lag_pvals[i] < 0.05]
            if anlamli:
                en_guclu = max(anlamli, key=lambda x: abs(x[1]))
                yon = "negatif" if en_guclu[1] < 0 else "pozitif"
                yorum = (f"En güçlü anlamlı sinyal **+{en_guclu[0]} günlük** gecikmede "
                         f"(r = {en_guclu[1]:.3f}, {yon} korelasyon). ")
                if en_guclu[1] < 0 and rm != "Hacim":
                    yorum += f"Yüksek {rm_label} bugün → gelecekte **düşük getiri** eğilimi."
                elif en_guclu[1] > 0 and rm != "Hacim":
                    yorum += f"Yüksek {rm_label} bugün → gelecekte **yüksek getiri** eğilimi (dikkatli yorumla)."
                elif rm == "Hacim":
                    yorum += "Yüksek hacim bugün → " + ("gelecekte pozitif getiri eğilimi." if en_guclu[1] > 0 else "gelecekte negatif getiri eğilimi.")
                st.markdown(yorum)
            else:
                st.markdown(f"Seçilen boyut ({rm_label}) için istatistiksel olarak anlamlı lead-lag ilişkisi bulunamadı.")

            st.markdown("---")
            excel_df = metrics.iloc[::-1].copy()
            excel_df.index.name = "Date"
            excel_df = excel_df.reset_index()
            excel_df["Date"] = excel_df["Date"].dt.strftime("%d.%m.%Y")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                excel_df.to_excel(writer, index=False, sheet_name=_ticker)
            st.download_button(label="📥 Excel İndir (Tüm Veri)", data=buf.getvalue(), file_name=f"{_ticker}_{newest.replace('.','')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("👈 Soldaki konsoldan bir mod seçin ve **⚡ Başlat** butonuna tıklayın.")
    st.markdown("""
    ### Tablodaki Göstergeler

    | Gösterge | Açıklama |
    |---|---|
    | **Günlük Değ. (%)** | Önceki kapanışa göre değişim |
    | **Güniçi Değ. (%)** | (Kapanış − Açılış) / Açılış × 100 |
    | **Daily Range (₺)** | Yüksek − Düşük (mutlak fark) |
    | **Amihud (×10⁶)** | \|Getiri\| / Hacim × 10⁶ — düşük = likit |
    """)
