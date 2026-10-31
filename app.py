
import streamlit as st
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from rembg import remove
from ultralytics import YOLO


# ============================================================
# CONFIGURAÇÕES
# ============================================================

st.set_page_config(
    page_title="Severidade de giberela",
    page_icon="🌾",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "best.pt"
CLASS_ID_DOENCA = 0


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Ocultar o selo do Streamlit */
    [data-testid="stViewerBadge"],
    [class*="viewerBadge"],
    [class*="ViewerBadge"] {
        display: none !important;
    }

    /* Título principal */
    h1 {
        font-size: 36px !important;
    }

    /* Largura máxima e centralização da página */
    .stMainBlockContainer {
        max-width: 1100px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Ajuste para janelas estreitas */
    @media (max-width: 700px) {
        .stMainBlockContainer {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        h1 {
            font-size: 28px !important;
        }
    }

    /* Centralizar imagens */
    [data-testid="stImage"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CABEÇALHO
# ============================================================

st.markdown(
    '<h1>🌾 Avaliação da severidade de giberela em trigo</h1>',
    unsafe_allow_html=True
)

st.write(
    "Envie uma imagem já recortada da espiga. "
    "O aplicativo removerá o fundo, identificará a região "
    "doente por segmentação e calculará a severidade estimada."
)


# ============================================================
# CONFIGURAÇÕES AVANÇADAS
# ============================================================

with st.sidebar:
    with st.expander(
        "⚙️ Configurações avançadas",
        expanded=False
    ):

        conf = st.slider(
            "Confiança mínima",
            min_value=0.05,
            max_value=0.95,
            value=0.30,
            step=0.05,
            format="%.2f",
            help=(
                "Confiança mínima para o modelo identificar "
                "uma área com giberela. Uma confiança menor "
                "pode detectar mais regiões, mas também pode "
                "aumentar os falsos positivos."
            )
        )

        imgsz = st.select_slider(
            "Tamanho de inferência",
            options=[320, 640, 800, 960, 1024],
            value=800,
            help=(
                "Define a resolução para a qual a imagem é "
                "redimensionada antes de ser processada pelo modelo."
                "Resoluções maiores podem preservar mais detalhes, "
                "mas nem sempre garantem maior precisão."
            )
        )

        limiar_planta = st.slider(
            "Limiar de remoção do fundo",
            min_value=0,
            max_value=255,
            value=10,
            step=1,
            help=(
                "Define o valor mínimo da transparência "
                "para considerar um pixel como parte da espiga. "
                "Valores maiores tornam a máscara mais restritiva."
            )
        )


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

@st.cache_resource
def carregar_modelo(caminho):
    return YOLO(str(caminho))


# ============================================================
# REMOÇÃO DO FUNDO
# ============================================================

def remover_fundo(imagem_pil, limiar_planta):
    resultado = remove(imagem_pil.convert("RGBA"))

    if isinstance(resultado, Image.Image):
        rgba = resultado.convert("RGBA")
    else:
        rgba = Image.open(BytesIO(resultado)).convert("RGBA")

    # Máscara da espiga baseada na transparência do rembg
    alpha = np.asarray(
        rgba.getchannel("A"),
        dtype=np.uint8
    )

    plant_mask = alpha > limiar_planta

    # Compor a imagem sobre fundo preto
    fundo = Image.new(
        "RGBA",
        rgba.size,
        (0, 0, 0, 255)
    )

    fundo.alpha_composite(rgba)
    imagem_sem_fundo = fundo.convert("RGB")

    return imagem_sem_fundo, plant_mask


# ============================================================
# SEGMENTAÇÃO DA GIBERELA
# ============================================================

def segmentar(modelo, imagem, confianca, imgsz):
    altura, largura = imagem.height, imagem.width

    resultado = modelo.predict(
        source=imagem,
        imgsz=imgsz,
        conf=confianca,
        retina_masks=True,
        save=False,
        verbose=False
    )[0]

    mascara_doenca = np.zeros(
        (altura, largura),
        dtype=bool
    )

    if resultado.masks is not None and resultado.boxes is not None:
        mascaras = resultado.masks.data.cpu().numpy()
        classes = (
            resultado.boxes.cls.cpu().numpy().astype(int)
        )

        for mascara, classe in zip(mascaras, classes):

            # Considerar somente a classe da doença
            if classe != CLASS_ID_DOENCA:
                continue

            # Garantir dimensões iguais às da imagem original
            if mascara.shape != (altura, largura):
                mascara_binaria = Image.fromarray(
                    (mascara >= 0.5).astype(np.uint8) * 255
                )

                mascara_binaria = mascara_binaria.resize(
                    (largura, altura),
                    Image.Resampling.NEAREST
                )

                mascara = np.asarray(mascara_binaria) > 0
            else:
                mascara = mascara >= 0.5

            mascara_doenca |= mascara

    return mascara_doenca, resultado


# ============================================================
# SOBREPOSIÇÃO DA REGIÃO DOENTE
# ============================================================

def overlay_doenca(imagem, mascara):
    base = np.asarray(
        imagem.convert("RGB")
    ).copy()

    vermelho = np.array(
        [255, 0, 0],
        dtype=np.float32
    )

    base[mascara] = (
        0.45 * vermelho
        + 0.55 * base[mascara].astype(np.float32)
    ).astype(np.uint8)

    return Image.fromarray(base)


# ============================================================
# UPLOAD DA IMAGEM
# ============================================================

arquivo = st.file_uploader(
    "Selecione a imagem recortada da espiga",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESSAMENTO
# ============================================================

if arquivo is not None:

    try:
        imagem_original = Image.open(arquivo).convert("RGB")

    except Exception:
        st.error("Não foi possível abrir a imagem enviada.")
        st.stop()

    if st.button(
        "Remover fundo e calcular severidade",
        type="primary",
        use_container_width=True
    ):

        if not MODEL_PATH.is_file():
            st.error(
                f"Checkpoint não encontrado: {MODEL_PATH}. "
                "Verifique se best.pt está no repositório."
            )
            st.stop()

        try:
            # ------------------------------------------------
            # 1. Remover fundo
            # ------------------------------------------------

            with st.spinner("Removendo o fundo da imagem..."):
                imagem_sem_fundo, plant_mask = remover_fundo(
                    imagem_original,
                    limiar_planta
                )

            area_total = int(plant_mask.sum())

            if area_total == 0:
                st.error(
                    "Nenhum pixel da espiga foi identificado. "
                    "Tente outra imagem ou ajuste o limiar "
                    "nas configurações avançadas."
                )
                st.stop()

            # ------------------------------------------------
            # 2. Segmentar região doente
            # ------------------------------------------------

            with st.spinner(
                "Identificando a região com giberela..."
            ):
                modelo = carregar_modelo(MODEL_PATH)

                disease_mask, resultado = segmentar(
                    modelo,
                    imagem_sem_fundo,
                    conf,
                    imgsz
                )

            # Restringir a doença à área identificada como espiga
            disease_mask &= plant_mask

            area_doente = int(disease_mask.sum())

            # ------------------------------------------------
            # 3. Calcular severidade
            # ------------------------------------------------

            severidade = (
                100.0 * area_doente / area_total
            )

            # Garantir que o fundo permaneça preto
            imagem_limpa_np = np.asarray(
                imagem_sem_fundo
            ).copy()

            imagem_limpa_np[~plant_mask] = 0

            imagem_limpa = Image.fromarray(
                imagem_limpa_np
            )

            imagem_overlay = overlay_doenca(
                imagem_limpa,
                disease_mask
            )

            # ------------------------------------------------
            # 4. Apresentar resultados
            # ------------------------------------------------

            st.success("Processamento concluído!")
            st.subheader("Resultado")

            col_a, col_b, col_c = st.columns(3)

            col_a.metric(
                "Severidade estimada",
                f"{severidade:.2f}%"
            )

            col_b.metric(
                "Área da espiga",
                f"{area_total:,}".replace(",", ".")
                + " pixels"
            )

            col_c.metric(
                "Área doente",
                f"{area_doente:,}".replace(",", ".")
                + " pixels"
            )

            n_instancias = (
                len(resultado.boxes)
                if resultado.boxes is not None
                else 0
            )

            st.caption(
                f"Confiança: {conf:.2f} | "
                f"imgsz: {imgsz} | "
                f"Limiar do fundo: {limiar_planta} | "
                f"Instâncias detectadas: {n_instancias}"
            )

            st.markdown("### Visualização dos resultados")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    imagem_limpa,
                    caption="Espiga após remoção do fundo",
                    width=230
                )

            with col2:
                st.image(
                    imagem_overlay,
                    caption="Região identificada como doente",
                    width=230
                )

            # ------------------------------------------------
            # 5. Relatório para download
            # ------------------------------------------------

            relatorio = (
                "RELATÓRIO DE SEVERIDADE DE GIBERELA\n"
                "===================================\n\n"
                f"Imagem: {arquivo.name}\n"
                f"Dimensões: {imagem_original.width} × "
                f"{imagem_original.height} pixels\n"
                f"Área da espiga: {area_total} pixels\n"
                f"Área identificada como doente: "
                f"{area_doente} pixels\n"
                f"Severidade estimada: {severidade:.4f}%\n"
                f"Confiança mínima: {conf:.2f}\n"
                f"Tamanho de inferência: {imgsz}\n"
                f"Limiar de remoção do fundo: {limiar_planta}\n"
            )

            st.download_button(
                "Baixar relatório (.txt)",
                data=relatorio.encode("utf-8"),
                file_name=(
                    f"{Path(arquivo.name).stem}_severidade.txt"
                ),
                mime="text/plain"
            )

        except Exception as erro:
            st.error(
                "Ocorreu um erro durante o processamento."
            )
            st.exception(erro)

else:
    st.caption(
        "Envie uma imagem para iniciar a avaliação."
    )
