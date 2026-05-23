import requests
import streamlit as st


API_URL = "http://localhost:8000/api"


st.set_page_config(page_title="Handwriting OCR", page_icon="OCR", layout="wide")
st.title("Распознавание рукописного текста")


def api_get(path: str, timeout: int = 10):
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def render_system_status() -> None:
    with st.sidebar:
        st.header("Система")
        try:
            health = api_get("/health")
            models = api_get("/models")
            st.success(f"{health['app_name']} {health['version']}")
            for model in models:
                st.caption(f"Модель: {model['model_name']}")
                st.caption("Инференс: локальный")
        except requests.RequestException:
            st.error("Backend недоступен")


def render_recognition_tab() -> None:
    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file is None:
        st.info("Выберите изображение с рукописным текстом.")
        return

    left, right = st.columns([1, 1])
    with left:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        st.caption(f"Размер файла: {len(uploaded_file.getvalue()) / 1024:.1f} КБ")

    with right:
        if st.button("Распознать текст", type="primary", use_container_width=True):
            with st.spinner("Изображение обрабатывается локальной моделью..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/recognize", files=files, timeout=300)

            if response.ok:
                st.session_state["last_result"] = response.json()
            else:
                detail = response.json().get("detail", "Не удалось распознать изображение.")
                st.error(detail)

        result = st.session_state.get("last_result")
        if result:
            st.subheader("Результат")
            st.text_area("Распознанный текст", value=result["recognized_text"], height=220)
            st.download_button(
                "Экспорт в TXT",
                data=result["recognized_text"],
                file_name=f"recognition_{result['id']}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.caption(
                f"Модель: {result['model_name']} | "
                f"Время: {result['processing_time_ms']} мс | "
                f"Размер: {result['file_size_bytes'] / 1024:.1f} КБ"
            )


def render_history_tab() -> None:
    try:
        history = api_get("/history")
    except requests.RequestException:
        st.info("Запустите backend, чтобы увидеть историю распознаваний.")
        return

    if not history:
        st.info("История пока пуста.")
        return

    for item in history[:20]:
        with st.expander(f"{item['original_filename']} — {item['created_at']}"):
            image_left, image_right = st.columns(2)
            with image_left:
                st.caption("Исходное изображение")
                st.image(f"{API_URL.replace('/api', '')}{item['original_image_url']}", use_container_width=True)
            if item["processed_image_url"]:
                with image_right:
                    st.caption("После предобработки")
                    st.image(f"{API_URL.replace('/api', '')}{item['processed_image_url']}", use_container_width=True)

            st.text_area(
                "Распознанный текст",
                value=item["recognized_text"],
                height=160,
                key=f"history_text_{item['id']}",
            )
            st.caption(
                f"Модель: {item['model_name']} | "
                f"{item['processing_time_ms']} мс | "
                f"{item['file_size_bytes'] / 1024:.1f} КБ"
            )


def render_experiments_tab() -> None:
    st.subheader("Экспериментальная оценка")
    st.write(
        "Для дипломной части используется CSV-манифест с изображениями и эталонными текстами. "
        "Скрипт пакетно запускает OCR pipeline и считает CER/WER."
    )
    st.code(
        "python scripts/evaluate_dataset.py "
        "--manifest data/evaluation/manifest.csv "
        "--output data/evaluation/results.csv",
        language="bash",
    )
    st.caption("Шаблон манифеста: data/evaluation/manifest.example.csv")


render_system_status()
tab_recognition, tab_history, tab_experiments = st.tabs(["Распознавание", "История", "Эксперименты"])

with tab_recognition:
    render_recognition_tab()

with tab_history:
    render_history_tab()

with tab_experiments:
    render_experiments_tab()
