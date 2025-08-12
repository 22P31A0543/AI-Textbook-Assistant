import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

# -------------------------------
# Load models (smaller & faster)
# -------------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer, device=-1)  # CPU; change device=0 if GPU available

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📘 AI Textbook Assistant", layout="wide")
st.title("📘 AI Textbook Question Answering App (Fast CPU Version)")
st.markdown("Upload a textbook (PDF), ask any question, and get a smart AI answer — all without API credits!")

# -------------------------------
# PDF Upload & Parsing
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload your textbook (PDF)", type="pdf")

def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
    return chunks

if uploaded_file:
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    st.success("✅ PDF loaded successfully!")

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    embeddings = embedder.encode(paragraphs, convert_to_tensor=True)

    st.info(f"🔍 Indexed {len(paragraphs)} paragraphs for answering.")

    question = st.text_input("❓ Ask a question from the textbook")

    def ask_local_model(context, question):
        result = qa_model(question=question, context=context)
        return result["answer"]

    if st.button("🧠 Get Answer") and question:
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, embeddings, top_k=5)
        top_indices = hits[0]
        context = "\n\n".join([paragraphs[idx['corpus_id']] for idx in top_indices])
        answer = ask_local_model(context, question)
        st.success("✅ Answer:")
        st.write(answer)

    if st.button("📌 Summarize Entire Book"):
        st.info("⏳ Summarizing book...")
        chunks = chunk_text(text, max_tokens=512)
        full_summary = ""

        for chunk in tqdm(chunks, desc="Summarizing", leave=False):
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            full_summary += summary[0]['summary_text'] + "\n\n"

        st.success("✅ Summary Ready:")
        st.write(full_summary)
