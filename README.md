# Road-To-Transformers

I have started a road to Transformers, the mordern cornerstone for NLP tasks
So, my journey starts with a basics and leads up to advance topics in NLP:
It will go as follows:
1. N-gram model
2. Basic RNN
3. LSTM
4. Finally, Transformer

The goal here is to compare all the models, in terms of performance on the same dataset and evaluation metrics.
NOTE: I have already tried implementations of these, so I won't be dumbing them down mmuch, but will try to make the code as easy to interpret as possible

Right now, i dont have a blog for this series, but i do plan on making one, once i have uploaded all the code on this repo, so keep checking back:

---
### How to use the code:
- I will add a depedencies or env config on completion, so you need that
- But, for now torch is more that enough
- To run the code, just look at [test.py](/test.py)
- How does it works:
  - It picks a [models.py](/models.py), which has all the different models of the journey
  - It processes and takes data from class Data in [pre_process_data.py](/pre_process_data.py)
  - Then uses functions from [train_eval.py](/train_eval.py), to train that model and saves the weights

---
