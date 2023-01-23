import gradio as gr
from simplet5 import SimpleT5
import spacy
import pytextrank

def extract_important_sentences(text, limit_phrases=15, limit_sentences=5):
    en_nlp = spacy.load("en_core_web_sm")
    en_nlp.add_pipe("textrank", last=True)
    doc = en_nlp(text)
    tr = doc._.textrank
    summary = ""
    for sent in tr.summary(limit_phrases=limit_phrases, limit_sentences=limit_sentences):
        summary += sent.text + " "
    return summary

def create_summaries(text):

    # print("ACTUAL ABSTRACT - " + text)
    # print("\nLength of Abstract = " + str(len(text.split())))
    sumtext = "summarize: " + text
    actual_text_prediction = model.predict(sumtext)[0]
    # print("\nDIRECT SUMMARIZATION USING T5 - " + actual_text_prediction)
    # print("\nLength of Summary = " + str(len(actual_text_prediction.split())))

    newtext = extract_important_sentences(text, 20, 6)
    newtext = "summarize: " + newtext
    extractive_text_prediction = model.predict(newtext)[0]
    # print("\nSUMMARIZATION AFTER EXTRACTIVE USING T5 - " + extractive_text_prediction)
    # print("\nLength of Summary = " + str(len(extractive_text_prediction.split())))
    
    return actual_text_prediction, extractive_text_prediction
    
model = SimpleT5()
model.load_model("t5", "simplet5-epoch-9-train-loss-1.3675-val-loss-2.6217")

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            inputBox = gr.TextArea(label="Enter Abstract of any Research Paper")
            b1 = gr.Button("Perform Summarization")
    with gr.Row():
        text1 = gr.TextArea(label="Direct T5 Summary")
        text2 = gr.TextArea(label="After Extractive T5 Summary")

    b1.click(create_summaries, inputs=inputBox, outputs=[text1, text2])
    
# Running the app
iface.launch()