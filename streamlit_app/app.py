import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import os

# Configure page settings
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Constants
MODEL_PATH = "../Fine-Tuned-BERT-for-Movie-Review-Classification-with-PEFT-and-LoRA/distilbert-base-uncased-lora-text-classification/checkpoint-125"
BASE_MODEL = "distilbert-base-uncased"
ID2LABEL = {0: "Negative", 1: "Positive"}

@st.cache_resource
def load_model_and_tokenizer():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load base model with PEFT configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL
    )
    
    # Load and apply PEFT/LoRA adapters
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model.eval()
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1)
        confidence = probabilities[0][prediction].item()
    
    return ID2LABEL[prediction.item()], confidence

# Page title and description
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("""
This app analyzes the sentiment of movie reviews using a fine-tuned BERT model.
Enter your review below to find out if it expresses a positive or negative sentiment!
""")

# Load model and tokenizer
try:
    model, tokenizer = load_model_and_tokenizer()
    st.success("Model loaded successfully! Ready to analyze reviews.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Text input area
review_text = st.text_area(
    "Enter your movie review:",
    height=150,
    placeholder="Type or paste your movie review here..."
)

# Example reviews
with st.expander("Click to see example reviews"):
    st.markdown("""
    - *"The movie was an absolute delight from start to finish!"*
    - *"I wouldn't recommend this film to anyone; it was quite a letdown."*
    - *"A truly captivating experience with stunning performances and a gripping plot."*
    - *"This film is a waste of time; it fails to engage or entertain."*
    """)

# Analyze button
if st.button("Analyze Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                sentiment, confidence = predict_sentiment(review_text, model, tokenizer)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Sentiment")
                    sentiment_color = "green" if sentiment == "Positive" else "red"
                    st.markdown(f"<h2 style='color: {sentiment_color};'>{sentiment}</h2>", 
                              unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Confidence")
                    st.progress(confidence)
                    st.text(f"{confidence*100:.1f}%")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

st.markdown("---")
st.markdown("""
### How to use:
1. Type or paste your movie review in the text box above
2. Click the "Analyze Sentiment" button
3. View the predicted sentiment and confidence score

The model will classify your review as either Positive or Negative and show how confident it is about the prediction.
""")