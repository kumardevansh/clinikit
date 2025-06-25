import gradio as gr
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import faiss
import json
import time
import os
from dotenv import load_dotenv
import speech_recognition as sr

# Load resources
index = faiss.read_index("health_index.faiss")
with open("health_meta.json") as f:
    data = json.load(f)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Set your Gemini API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API")
genai.configure(api_key=API_KEY)  # Replace with your actual API key

# Initialize the Gemini model
gen_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')

def get_health_response(user_query: str, user_query_org: str, k=3):
    query_emb = model.encode([user_query], convert_to_numpy=True)
    _, indices = index.search(query_emb, k=k)
    
    context = "\n".join([f"Disease: {data[idx]['disease']}\nSymptoms: {data[idx]['symptoms']}"
                        f"\nDescription: {data[idx]['description']}" 
                        for idx in indices[0]])
    
    prompt = f"""You are a medical health assistant which answers health related queries or tells the disease based on the symptoms provided
For the given context: {context}
Answer this question in the same language as the question: {user_query_org}
and cite a trustworthy source (like healthline, WebMD, wikipedia or WHO).
Also if you recieve non-medical queries, tell the user to ask only health related queries.
Answer:"""
    
    response = gen_model.generate_content(prompt)
    
    return response.text.strip()

def generate_translation(text):
    prompt = f"Give only the most accurate English translation of the given text if it is any other language except English, if the input is already in English return it as it is, nothing else: {text}"
    try:
        response = gen_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error for {text}")
        return "Description not available."

from gradio.themes.utils import fonts
from gradio.themes.base import Base

class HealthTheme(Base):
    def __init__(self):
        super().__init__(
            font=[
                fonts.GoogleFont("Asap"),
                fonts.Font("ui-sans-serif"),
                fonts.Font("sans-serif")
            ],
            font_mono=[
                fonts.GoogleFont("Fira Code"),
                fonts.Font("ui-monospace"),
                fonts.Font("monospace")
            ]
        )

        self.set(
            body_background_fill="#FFFFFF",
            body_background_fill_dark="linear-gradient(to right, #001027, #00112e, #001235, #00123c, #001142)",
            background_fill_primary="#FFFFFF",
            background_fill_primary_dark="#19191956",
            background_fill_secondary="#ECF2F7",
            background_fill_secondary_dark="linear-gradient(to right, #000b1a, #000b1e, #000b22, #000b26, #000b2a)",
            block_background_fill="#ECF2F7",
            block_background_fill_dark="linear-gradient(to right, #000b1a, #000b1e, #000b22, #000b26, #000b2a)",
            block_border_color="#dce3e8",
            block_border_color_dark="#000431",
            button_primary_background_fill="#338AC9",
            button_primary_background_fill_dark="#0c6ebd", 
            button_primary_background_fill_hover="#0c6ebd",
            button_primary_background_fill_hover_dark="#000538",
            button_primary_text_color="#ECF2F7",
            button_primary_text_color_dark="#08003BFF",
            button_primary_text_color_hover_dark="#ECF2F7",
            input_background_fill="#dce3e8",
            input_background_fill_dark="#FF0000FF",
            block_label_text_color="#4EACEF",
            block_label_text_color_dark="#4EACEF",
            block_title_text_color="#4EACEF",
            loader_color="#4EACEF",
            loader_color_dark="#4EACEF",
            body_text_color="#191919",
            body_text_color_dark="#ECF2F7",
            body_text_color_subdued="#636668",
            body_text_color_subdued_dark="#c4c4c4",
            body_text_size="*text_md",
            body_text_weight="400",
            border_color_accent="#dce3e8",
            border_color_accent_dark="#242424",
            border_color_accent_subdued="#dce3e867",
            border_color_accent_subdued_dark="#24242467",
            border_color_primary="#dce3e8",
            border_color_primary_dark="#242424",
            button_border_width="*input_border_width",
            button_border_width_dark="*input_border_width",
            button_cancel_background_fill="#dce3e8",
            button_cancel_background_fill_dark="#242424",
            button_cancel_background_fill_hover="#d0d7db",
            button_cancel_background_fill_hover_dark="#202020",
            button_cancel_border_color="#191919",
            button_cancel_border_color_dark="#ECF2F7",
            button_cancel_border_color_hover="#202020",
            button_cancel_border_color_hover_dark="#a1c3d8",
            button_cancel_text_color="#4EACEF",
            button_cancel_text_color_dark="#4EACEF",
            button_cancel_text_color_hover="#0c6ebd",
            button_cancel_text_color_hover_dark="#0c6ebd",
            button_large_padding="*spacing_lg calc(2 * *spacing_lg)",
            button_large_radius="*radius_lg",
            button_large_text_size="*text_lg",
            button_large_text_weight="600",
            button_primary_border_color="#191919",
            button_primary_border_color_dark="#ECF2F7",
            button_primary_border_color_hover="#202020",
            button_primary_border_color_hover_dark="#a1c3d8",
            button_primary_text_color_hover="#e1eaf0",
            button_secondary_background_fill="#dce3e8",
            button_secondary_background_fill_dark="#040052",
            button_secondary_background_fill_hover="#d0d7db",
            button_secondary_background_fill_hover_dark="#000644",
            button_secondary_border_color="#dce3e8",
            button_secondary_border_color_dark="#242424",
            button_secondary_border_color_hover="#d0d7db",
            button_secondary_border_color_hover_dark="#202020",
            button_secondary_text_color ="#4EACEF",
            button_secondary_text_color_dark="#4EACEF",
            button_secondary_text_color_hover="#0c6ebd",
            button_secondary_text_color_hover_dark="#d9eeff",
            button_small_padding="*spacing_sm calc(2 * *spacing_sm)",
            button_small_radius ="*radius_lg",
            button_small_text_size="*text_md",
            button_small_text_weight="400",
            button_transition   ="background-color 0.2s ease",
            color_accent="*primary_500",
            color_accent_soft="#dce3e8",
            color_accent_soft_dark="#0e1834",
            container_radius="*radius_lg",
            embed_radius="*radius_lg",
            error_background_fill="#dce3e8",
            error_background_fill_dark="#242424",
            error_border_color="#191919",
            error_border_color_dark="#ECF2F7",
            error_border_width="1px",
            error_border_width_dark="1px",
            error_icon_color="#b91c1c",
            error_icon_color_dark="#ef4444",
            error_text_color="#4EACEF",
            error_text_color_dark="#4EACEF",
            form_gap_width="0px",
            input_background_fill_focus="#dce3e8",
            input_background_fill_focus_dark="#2F2626",
            input_background_fill_hover="#d0d7db",
            input_background_fill_hover_dark="#202020",
            input_border_color="#191919",
            input_border_color_dark="#ECF2F7",
            input_border_color_focus="#191919",
            input_border_color_focus_dark="#ECF2F7",
            input_border_color_hover="#202020",
            input_border_color_hover_dark="#a1c3d8",
            input_border_width="0px",
            input_padding="*spacing_xl",
            input_placeholder_color="#19191930",
            input_placeholder_color_dark="#FFFFFF4F",
            input_radius="*radius_lg",
            input_shadow="#19191900",
            input_shadow_dark="#ECF2F700",
            input_shadow_focus="#19191900",
            input_shadow_focus_dark="#ECF2F700",
            input_text_size="*text_md",
            input_text_weight="400",
            layout_gap="*spacing_xxl",
            link_text_color="#4EACEF",
            link_text_color_active="#4EACEF",
            link_text_color_active_dark="#4EACEF",
            link_text_color_dark        ="#4EACEF",
            link_text_color_hover       ="#0c6ebd",
            link_text_color_hover_dark="#0c6ebd",
            link_text_color_visited     ="#4EACEF",
            link_text_color_visited_dark="#4EACEF",
        )

# Use the theme
custom_theme = HealthTheme()

# Add audio transcription function
def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "[Could not understand audio]"
        except sr.RequestError:
            return "[Audio service error]"

def print_like_dislike(x: gr.LikeData):
    print(f"Message #{x.index} was {'liked' if x.liked else 'disliked'}: {x.value}")

# Modified add_message function
def add_message(history, message):
    # Process text input
    if message["text"]:
        history.append({"role": "user", "content": message["text"]})
    
    # Process files (including audio)
    if message.get("files"):
        for file in message["files"]:
            # Transcribe audio files
            if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                transcribed = transcribe_audio(file)
                history.append({"role": "user", "content": f"[Audio]: {transcribed}"})
            # Handle other file types
            else:
                history.append({"role": "user", "content": f"[File received: {file}]"})
    
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    # Context window of last N turns
    N = 6
    memory_context = ""
    for turn in history[-N:]:
        if isinstance(turn["content"], str):
            role = turn["role"]
            prefix = "User" if role == "user" else "Assistant"
            memory_context += f"{prefix}: {turn['content']}\n"

    user_input = history[-1]["content"]
    translated = generate_translation(user_input)
    full_prompt = f"{memory_context}User: {translated}\nAssistant:"

    response = get_health_response(full_prompt, user_input)

    history.append({"role": "assistant", "content": ""})
    for char in response:
        history[-1]['content'] += char
        time.sleep(0.02)
        yield history

def undo(history):
    if len(history) >= 2:
        return history[:-2]
    return []

def retry(history):
    if len(history) >= 2:
        last_user = history[-2]
        history = history[:-2] + [last_user]
        return history
    return history

# --- UI Setup ---
with gr.Blocks(theme = custom_theme) as demo:
    gr.Markdown("""<h1 style='font-weight:600;'>🩺 Clinikit</h1>
<p style='color:#666;font-size:15px'>Ask health-related questions or enter symptoms below. Built with memory, streaming, multilingual text support and voice inputs(english only).</p>""")

    chatbot = gr.Chatbot(
        label="Assistant",
        type="messages",
        avatar_images=(None, "https://e7.pngegg.com/pngimages/369/865/png-clipart-physician-hospital-dr-mary-c-kirk-md-doctor-of-medicine-computer-icons-the-doctor-miscellaneous-black-thumbnail.png")
    )
    msg = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter symptoms or ask a health question...",
        show_label=False,
        sources=["microphone"]
    )
    with gr.Row():
        retry_btn = gr.Button("🔁 Retry")
        undo_btn = gr.Button("↩️ Undo")

    chat_msg = msg.submit(add_message, [chatbot, msg], [chatbot, msg])
    bot_msg = chat_msg.then(bot, chatbot, chatbot)
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])

    retry_btn.click(retry, chatbot, chatbot).then(bot, chatbot, chatbot).then(lambda h: h, chatbot, chatbot)
    undo_btn.click(undo, chatbot, chatbot).then(lambda h: h, chatbot, chatbot)

    chatbot.like(print_like_dislike, None, None, like_user_message=True)

    gr.Markdown("""<footer style='text-align:center;margin-top:20px;color:#aaa;'>Built using Gradio, Hugging Face & Gemini API</footer>""")

demo.launch(share=True, pwa=True)
