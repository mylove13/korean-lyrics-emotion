
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 모델 불러오기 (긍정/부정 감정 분류용)
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 감정 분석 함수
def analyze_korean_sentiment(text):
    result = classifier(text)[0]
    label_kr = "긍정" if result["label"] == "LABEL_1" else "부정"
    return f"예측 감정: {label_kr} ({result['score']:.2%} 확신)"

# Gradio 인터페이스
gr.Interface(
    fn=analyze_korean_sentiment,
    inputs=gr.Textbox(label="한글 노래 가사 입력"),
    outputs="text",
    title="🎵 한글 노래 가사 감정 분석기",
    description="한글 노래 가사 또는 문장을 입력하면 감정을 예측해드립니다 (긍정/부정)"
).launch()
