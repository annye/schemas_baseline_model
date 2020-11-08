from pdb import set_trace
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from random import choice
import panel as pn
import panel.widgets as pnw
import random
pn.extension() # loading panel's extension for jupyter compatibility 
text_input = pn.widgets.TextInput()

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
def get_pred(text, model, tok, p=0.7):
    input_ids = torch.tensor(tok.encode(text)).unsqueeze(0)
    logits = model(input_ids)[0][:, -1]
    probs = F.softmax(logits, dim=-1).squeeze()
    idxs = torch.argsort(probs, descending=True)
    res, cumsum = [], 0.
    for idx in idxs:
        res.append(idx)
        cumsum += probs[idx]
        if cumsum > p:
            pred_idx = idxs.new_tensor([choice(res)])
            break
    pred = tok.convert_ids_to_tokens(int(pred_idx))
    return tok.convert_tokens_to_string(pred)

get_pred("where is the cat",model,tok, p=0.7)

text_input = pn.widgets.TextInput()
generated_text = pn.pane.Markdown(object=text_input.value)
text_input.link(generated_text, value='object')


text_input = pn.widgets.TextInput(width=1400)
text_input.link(generated_text, value='object')
text_input.link(generated_text, value='object')

button = pn.widgets.Button(name="Generate",button_type="primary")
def click_cb(event):
    pred = get_pred(generated_text.object, model, tok)
    generated_text.object += pred

button.on_click(click_cb)
app = pn.Column(text_input, button, generated_text); app



text2_input = pn.widgets.TextInput()


text = """

     
     
     
     
     
     
     
     
     
     
     
     
     
   
   
"""












radio_group2 = pn.widgets.RadioButtonGroup(
    name='Radio Button Group', options=['Rhetorical Devices', 'Cialdini Principles', 'Cognitive-Emotional'], button_type='primary')


radio_group = pn.widgets.RadioButtonGroup(
    name='Radio Button Group', options=['#Vaccination', '#High Level Education', '#Computer Science'], button_type='default')





title = pn.pane.Markdown("# **Text Generator**")

desc = pn.pane.HTML("<marquee scrollamount='10'><b>Welcome: Write key words topic! In order to get started, simply enter some starting  input text below, click generate a few times and watch it go!</b></marquee>")

# final_app = pn.Column(
#     text, 
#     title, desc ,app)

final_app = pn.Column(
    '#-------------------------------------------------------- Persuasive Style Schemas-------------------------------------------------------------------',
#     '####Select Persuasive Style Schema ',
   
    pn.layout.Divider(),
   
    radio_group2,radio_group,desc,app,text,
    background='whitesmoke', width_policy='max'
)

final_app.show()





