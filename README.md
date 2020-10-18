# Neural-Machine-Translation

The idea of this project was to implement seq2seq using Encoder-Decoder Architecture using GRU's.
The dataset used here was from https://nlp.stanford.edu/projects/nmt/ .
I have used English to Vietnamese dataset.
If you want to train the model yourself, I would suggest you to use a GPU or use google colab to do so.


## Testing can be done using the command:

``` python
python NMT.py test
```
## The translation command is:
```python
python NMT.py translate
```
## Conclusion
#### The average BLEU score over the entire test data set is 19.35 which is very good
