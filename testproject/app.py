# -----------------------------------------------------------------------------
# app.py – Streamlit web UI (run: `streamlit run app.py`)
# -----------------------------------------------------------------------------
import streamlit as st
from PIL import Image
from io import BytesIO

from inference import load_scoring_model, predict_happy_score
from generator import make_happier_face

st.set_page_config(page_title="Happyscore", page_icon="😊", layout="centered")
st.title("😊 Happyscore")

with st.expander("About this demo"):
    st.markdown(
        """
        **Happyscore** predicts how happy a face looks (1–10).  
        Click **Upgrade happiness** to generate a happier version using Stable Diffusion.
        """
    )

upload = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
if upload:
    from io import BytesIO
    img = Image.open(BytesIO(upload.read())).convert("RGB")
    st.image(img, caption="Original", width=256)

    if st.button("Get Happy Score 🎯"):
        model, device = load_scoring_model()
        score = predict_happy_score(img, model, device="mps")
        st.metric("Happy Score", f"{score}/10")

    if st.button("Upgrade happiness ✨"):
        with st.spinner("Generating happier face…"):
            small = img.resize((512, 512))
            upgraded = make_happier_face(img)
            st.image(upgraded, caption="Happier!", width=256)
            st.success("Done!")
