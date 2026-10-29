import streamlit as st

st.markdown(
    """
    <style>
    [data-testid="stViewerBadge"],
    [class*="viewerBadge"],
    [class*="ViewerBadge"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from ultralytics import YOLO

MODEL_PATH = Path("best.pt")
CLASS_ID_DOENCA = 0
LIMIAR_PLANTA = 10

st.set_page_config(page_title="Severidade de giberela", page_icon="🌾", layout="wide")
st.markdown(
    '<h1 style="font-size: 36px;"> 🌾 Avaliação da severidade de giberela em trigo</h1>',
    unsafe_allow_html=True
)
st.write("Envie uma imagem já recortada da espiga. O aplicativo removerá o fundo, segmentará a região identificada como doente e calculará a severidade.")
@st.cache_resource
def carregar_modelo(caminho):
    return YOLO(caminho)

def png_bytes(imagem):
    buffer = BytesIO()
    imagem.save(buffer, format="PNG")
    return buffer.getvalue()

def remover_fundo(imagem_pil):
    rgba = remove(imagem_pil.convert("RGBA"))
    if not isinstance(rgba, Image.Image):
        rgba = Image.open(BytesIO(rgba)).convert("RGBA")
    else:
        rgba = rgba.convert("RGBA")
    # A transparência produzida pelo rembg define a área total da espiga.
    plant_mask = np.asarray(rgba.getchannel("A"), dtype=np.uint8) > LIMIAR_PLANTA
    fundo = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    fundo.alpha_composite(rgba)
    return fundo.convert("RGB"), plant_mask

def segmentar(modelo, imagem, confianca, imgsz):
    rgb = np.asarray(imagem.convert("RGB"))
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    r = modelo.predict(source=bgr, imgsz=imgsz, conf=confianca,
                       retina_masks=True, save=False, verbose=False)[0]
    mask_doenca = np.zeros((h, w), dtype=bool)
    if r.masks is not None and r.boxes is not None:
        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for mask, classe in zip(masks, classes):
            if classe != CLASS_ID_DOENCA:
                continue
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_LINEAR)
            mask_doenca |= mask >= 0.5
    return mask_doenca, r

def overlay_doenca(imagem, mascara):
    base = np.asarray(imagem.convert("RGB")).copy()
    base[mascara] = (0.45 * np.array([255, 0, 0]) + 0.55 * base[mascara]).astype(np.uint8)
    return Image.fromarray(base)

def mascara_png(mascara):
    return Image.fromarray(mascara.astype(np.uint8) * 255, mode="L")

with st.sidebar:
    st.header("Parâmetros")
    conf = st.slider("Confiança mínima", 0.05, 0.95, 0.30, 0.05)
    imgsz = st.select_slider("Tamanho de inferência", options=[320, 640, 800, 960, 1024], value=640)


arquivo = st.file_uploader("Selecione a imagem recortada", type=["jpg", "jpeg", "png"])

if arquivo is not None:
    imagem_original = Image.open(arquivo).convert("RGB")
    if st.button("Remover fundo e calcular severidade", type="primary", use_container_width=True):
        if not MODEL_PATH.exists():
            st.error(f"Checkpoint não encontrado: {MODEL_PATH}.")
            st.stop()
        try:
            with st.spinner("Removendo o fundo..."):
                imagem_sem_fundo, plant_mask = remover_fundo(imagem_original)
            area_total = int(plant_mask.sum())
            if area_total == 0:
                st.error("Nenhum pixel da espiga foi identificado pelo removedor de fundo.")
                st.stop()
            with st.spinner("Segmentando a região doente..."):
                modelo = carregar_modelo(str(MODEL_PATH))
                disease_mask, resultado = segmentar(modelo, imagem_sem_fundo, conf, imgsz)
            disease_mask &= plant_mask
            area_doente = int(disease_mask.sum())
            severidade = 100.0 * area_doente / area_total
            imagem_limpa_np = np.asarray(imagem_sem_fundo).copy()
            imagem_limpa_np[~plant_mask] = 0
            imagem_limpa = Image.fromarray(imagem_limpa_np)
            mascara = mascara_png(disease_mask)
            overlay = overlay_doenca(imagem_limpa, disease_mask)

            st.subheader("Resultado")
            a, b, c = st.columns(3)
            a.metric("Severidade estimada", f"{severidade:.2f}%")
            b.metric("Área da espiga", f"{area_total:,}".replace(",", ".") + " pixels")
            c.metric("Área doente", f"{area_doente:,}".replace(",", ".") + " pixels")

            st.markdown(
                """
                <style>
                [data-testid="stImage"] {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(imagem_limpa, caption="Fundo removido", width=200)
            with col2:
                st.image(overlay, caption="Região doente destacada em vermelho", width=200)

            relatorio = (
                "Relatório de severidade de giberela\n"
                f"Imagem: {arquivo.name}\n"
                f"Dimensões: {imagem_original.width} × {imagem_original.height} pixels\n"
                f"Área da espiga: {area_total} pixels\n"
                f"Área identificada como doente: {area_doente} pixels\n"
                f"Severidade estimada: {severidade:.4f}%\n"
                f"Confiança: {conf:.2f}\nimgsz: {imgsz}\n"
            )
            st.download_button("Baixar relatório (.txt)", relatorio.encode("utf-8"),
                               file_name=f"{Path(arquivo.name).stem}_severidade.txt", mime="text/plain")

        except Exception as erro:
            st.exception(erro)
else:
    st.caption("Envie uma imagem para iniciar.")