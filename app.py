import gradio as gr
from simplet5 import SimpleT5
import spacy
import pytextrank
from eaas import Config, Client

def extract_important_sentences(text, limit_phrases=15, limit_sentences=5):
    en_nlp = spacy.load("en_core_web_sm")
    en_nlp.add_pipe("textrank", last=True)
    doc = en_nlp(text)
    tr = doc._.textrank
    summary = ""
    for sent in tr.summary(limit_phrases=limit_phrases, limit_sentences=limit_sentences):
        summary += sent.text + " "
    return summary

model = SimpleT5()
model.load_model("t5", "simplet5-epoch-9-train-loss-1.3675-val-loss-2.6217")

def use_gradio_summary(text):

    sumtext = "summarize: " + text
    actual_text_prediction = model.predict(sumtext)[0]

    newtext = extract_important_sentences(text, 20, 6)
    newtext = "summarize: " + newtext
    extractive_text_prediction = model.predict(newtext)[0]
    
    return actual_text_prediction, extractive_text_prediction
    

def calculate_scores(src, text):
    client = Client(Config())
    metrics = ["bert_score_f"]

    inputs = [{
        "references":src,
        "hypothesis":text
    }]

    score_dic = client.score(inputs, metrics=metrics)
    
    return str(float(score_dic['scores'][0]['corpus']) * 100)[:5]

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            inputBox = gr.TextArea(label="Enter Abstract of any Research Paper")
            b1 = gr.Button("Perform Summarization")
    with gr.Row():
        text1 = gr.TextArea(label="Direct T5 Summary")
        text2 = gr.TextArea(label="After Extractive T5 Summary")
    with gr.Row():
      with gr.Column():
        text1score = gr.Textbox(label="BERT Score Direct T5")
        bleft = gr.Button("Evaluate Direct T5")
      with gr.Column():
        text2score = gr.Textbox(label="BERT Score After Extractive T5 Summary")
        bright = gr.Button("Evaluate Extractive + T5")        
    
# Running the app
iface.launch()
