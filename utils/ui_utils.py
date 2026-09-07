import streamlit as st

COLOR_PRIMARY = "#D0001B"
COLOR_DARK_RED = "#A80016"
COLOR_WHITE = "#FFFFFF"
COLOR_GRAY = "#F7F7F7"
COLOR_TEXT = "#222222"


def inject_custom_css():
    """Injiziert das benutzerdefinierte Design im Stil des Katharineums."""
    css = f"""
    <style>
        /* Globales Styling */
        .stApp {{
            background-color: {COLOR_GRAY};
            color: {COLOR_TEXT};
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }}
        
        /* Header-Bereich mit Diagonale */
        .kh-header {{
            background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_DARK_RED} 100%);
            color: white;
            padding: 2.5rem 2rem;
            border-radius: 0 0 25px 25px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(208, 0, 27, 0.2);
        }}
        
        .kh-header h1 {{
            font-size: 3.2rem;
            font-weight: 900;
            margin: 0;
            letter-spacing: 2px;
            text-transform: uppercase;
        }}
        
        .kh-header p {{
            font-size: 1.2rem;
            font-weight: 300;
            margin-top: 0.5rem;
            letter-spacing: 1px;
        }}

        /* Aktions-Karten (Home) */
        .kh-card {{
            background-color: {COLOR_WHITE};
            border: 2px solid {COLOR_PRIMARY};
            border-radius: 18px;
            padding: 2rem;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }}
        
        .kh-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(208, 0, 27, 0.15);
        }}

        .kh-card-icon {{
            font-size: 3.5rem;
            margin-bottom: 1rem;
        }}

        .kh-card h3 {{
            color: {COLOR_PRIMARY};
            font-weight: 800;
            font-size: 1.6rem;
            margin-bottom: 0.8rem;
        }}

        /* Custom Streamlit Buttons */
        .stButton>button {{
            background-color: {COLOR_PRIMARY};
            color: white !important;
            border-radius: 12px;
            border: none;
            font-weight: bold;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            background-color: {COLOR_DARK_RED};
            border-color: {COLOR_DARK_RED};
            box-shadow: 0 4px 12px rgba(168, 0, 22, 0.3);
        }}

        /* Status-Badges */
        .badge-available {{
            background-color: #28a745;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 50px;
            font-weight: bold;
            font-size: 0.85rem;
        }}
        
        .badge-reserved {{
            background-color: #ffc107;
            color: #222;
            padding: 0.3rem 0.8rem;
            border-radius: 50px;
            font-weight: bold;
            font-size: 0.85rem;
        }}

        /* Footer */
        .kh-footer {{
            text-align: center;
            padding: 2rem;
            margin-top: 4rem;
            border-top: 2px solid #FFE6E6;
            color: #666;
            font-weight: 500;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header():
    """Rendert den Katharineum Header."""
    st.markdown("""
        <div class="kh-header">
            <h1>FUNDBÜRO</h1>
            <p>KATHARINEUM ZU LÜBECK</p>
        </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Rendert den Footer."""
    st.markdown("""
        <div class="kh-footer">
            <p>© Katharineum zu Lübeck – Digitales KI-Fundbüro</p>
        </div>
    """, unsafe_allow_html=True)


def render_home_card(title: str, description: str, icon_name: str, button_key: str):
    """Rendert eine Hauptaktionskarte."""
    icon_symbol = "🔍" if icon_name == "search" else "🧺"
    st.markdown(f"""
        <div class="kh-card">
            <div class="kh-card-icon">{icon_symbol}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
    """, unsafe_allow_html=True)
    st.button(f"{title} ÖFFNEN", key=button_key, use_container_width=True)


def render_item_card(item: dict):
    """Rendert eine Übersichtskarte für ein Fundstück."""
    status = item.get("status", "Verfügbar")
    badge_class = "badge-available" if status == "Verfügbar" else "badge-reserved"
    
    st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 1rem; border: 1px solid #ddd; margin-bottom: 1rem;">
            <span class="{badge_class}">{status}</span>
            <h4 style="color: #D0001B; margin-top: 0.5rem;">{item.get('category')}</h4>
            <p><b>ID:</b> {item.get('id')}<br>
            <b>Farbe:</b> {item.get('color')}<br>
            <b>Fundort:</b> {item.get('location_found')}</p>
        </div>
    """, unsafe_allow_html=True)


def render_prediction_result(prediction: dict):
    """Zeigt KI-Ergebnisse übersichtlich an."""
    st.markdown("### 🤖 KI-Analyseergebnis")
    label = prediction.get("label")
    conf = prediction.get("confidence", 0.0) * 100

    st.info(f"Erkannte Kategorie: **{label}** (Sicherheit: **{conf:.1f}%**)")

    with st.expander("Top-3 Wahrscheinlichkeiten anzeigen"):
        for cat, prob in prediction.get("probabilities", {}).items():
            st.write(f"- **{cat}**: {prob*100:.1f}%")


def render_status_badge(status: str):
    badge_class = "badge-available" if status == "Verfügbar" else "badge-reserved"
    st.markdown(f'<span class="{badge_class}">{status}</span>', unsafe_allow_html=True)
