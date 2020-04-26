
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow.compat.v1 as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
tf.keras.backend.clear_session()
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sm = SmoothingFunction()
import sys

embedding_size = 128
state_size = 512

with open("train.en.txt",'rt',encoding='utf-8') as f:
  english = f.readlines()
f.close()

with open("train.vit.txt",'rt',encoding='utf-8') as f1:
  vitn = f1.readlines()
f1.close()

for i in range(0,len(vitn)):
    vitn[i] = "starttt  " + vitn[i] + " enddd"

num_words = 10000

class TokenizerWrap(Tokenizer):
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):

        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
    
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
          
            truncating = 'post'

        self.num_tokens = [len(x) for x in self.tokens]

        self.max_tokens = np.mean(self.num_tokens) \
                          + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
 
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):

        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            tokens = np.flip(tokens, axis=1)

            truncating = 'pre'
        else:

            truncating = 'post'

        if padding:
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)
        return tokens


tokenizer_eng = TokenizerWrap(texts=english,
                              padding='pre',
                              reverse=True,
                              num_words=num_words)     
 
tokenizer_vitn = TokenizerWrap(texts=vitn,
                               padding='post',
                               reverse=False,
                               num_words=num_words)

### This is to reduce the memory used for tokenizing the words.
tokens_eng = tokenizer_eng.tokens_padded
tokens_vitn = tokenizer_vitn.tokens_padded

encoder_input_data = tokens_eng

decoder_input_data = tokens_vitn[:, :-1]       # They are in reverse format, so reverse them
decoder_output_data = tokens_vitn[:, 1:]       # First 'start' marker is time stepped in output


##### Neural Network  #####
def encoder(embedding_size = 128 , state_size = 512):

    encoder_input = Input(shape=(None, ), name='encoder_input')

    # embedding_size = 128
    # state_size = 512

    encoder_embedding = Embedding(input_dim=num_words,
                                  output_dim=embedding_size,
                                  name='encoder_embedding')

    encoder_gru1 = GRU(state_size, name='encoder_gru1',
                       return_sequences=True)
    encoder_gru2 = GRU(state_size, name='encoder_gru2',
                       return_sequences=True)
    encoder_gru3 = GRU(state_size, name='encoder_gru3',
                       return_sequences=False)
    return encoder_input,encoder_embedding,encoder_gru1,encoder_gru2,encoder_gru3

encoder_input, encoder_embedding, encoder_gru1, encoder_gru2, encoder_gru3 = encoder(embedding_size = 128,state_size = 512)
def connect_encoder():

    nn = encoder_input
    nn = encoder_embedding(nn)

    nn = encoder_gru1(nn)
    nn = encoder_gru2(nn)
    nn = encoder_gru3(nn)

    encoder_output = nn
    
    return encoder_output

encoder_output = connect_encoder()


######### Similarly build the decoder
def decoder(embedding_size = 128,state_size = 512):
    decoder_initial_state = Input(shape=(state_size,),
                                  name='decoder_initial_state')

    decoder_input = Input(shape=(None, ), name='decoder_input')
    decoder_embedding = Embedding(input_dim=num_words,
                                  output_dim=embedding_size,
                                  name='decoder_embedding')

    decoder_gru1 = GRU(state_size, name='decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, name='decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, name='decoder_gru3',
                       return_sequences=True)

    decoder_dense = Dense(num_words, activation='linear', name='decoder_output')
    return decoder_initial_state ,decoder_input,decoder_embedding,decoder_gru1,decoder_gru2,decoder_gru3,decoder_dense

decoder_initial_state,decoder_input,decoder_embedding,decoder_gru1,decoder_gru2,decoder_gru3,decoder_dense= decoder(embedding_size = 128,state_size = 512)

def connect_decoder(initial_state):
    # Start the decoder-nnwork with its input-layer.
    nn = decoder_input

    # Connect the embedding-layer.
    nn = decoder_embedding(nn)
    
    # Connect all the GRU-layers.
    nn = decoder_gru1(nn, initial_state=initial_state)
    nn = decoder_gru2(nn, initial_state=initial_state)
    nn = decoder_gru3(nn, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(nn)
    
    return decoder_output

######## Now connect all layers and create the model.
    
decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])

decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])

def sparse_cross_entropy(y_true, y_pred):

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


optimizer = RMSprop(lr=1e-3)    ### Adam / Adagrad dosent work well with RNN's
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
model_train.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])

model_train.load_weights('nmt_train_model.h5')

def train():
    x_data = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data}

    y_data = {'decoder_output': decoder_output_data}
    validation_split = 10000 / len(encoder_input_data)
    print (validation_split)

    model_train.fit(x=x_data,
                    y=y_data,
                    batch_size=512,
                    epochs=10,
                    validation_split=validation_split,
                    )
mark_start = 'starttt'
mark_end = 'enddd'
token_start = tokenizer_vitn.word_index[mark_start.strip()]
token_end = tokenizer_vitn.word_index[mark_end.strip()]

model_train.save_weights('nmt_train_model.h5')
model_train.save('nmt_train_model.h5')

def translate(input_text,true_output_text = None):
    input_tokens = tokenizer_eng.text_to_tokens(text=input_text,reverse=True,padding=True)

    initial_state = model_encoder.predict(input_tokens)
    max_tokens = tokenizer_vitn.max_tokens

    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = {'decoder_initial_state': initial_state,'decoder_input': decoder_input_data}

        decoder_output = model_decoder.predict(x_data)

        token_onehot = decoder_output[0, count_tokens, :]

        token_int = np.argmax(token_onehot)
        if token_int != token_end:

            sampled_word = tokenizer_vitn.token_to_word(token_int)

            output_text += " " + sampled_word

            count_tokens += 1

    output_tokens = decoder_input_data[0]
    #print(input_text)
    print(output_text)
    return input_text, output_text, true_output_text


def test():
      ## Load the data
    file1 = open("tst2013.en.txt", encoding = "utf8")   # Load English Data
    english_test = file1.readlines()

    file2 = open("tst2013.vit.txt", encoding = "utf8")   # Load Vitnm Data
    vitn_test = file2.readlines()

    ### Now add a start and end marker for the destination language. 
    for i in range(0,len(vitn_test)):
        vitn_test[i] = "starttt " + vitn_test[i]
    scores_list = []
    for idx in range(0,len(english_test)):
        input_text, output_text, true_output_text = translate(input_text=english_test[idx],true_output_text=vitn_test[idx])
        #print(output_text)
        bleu = sentence_bleu([output_text], true_output_text, smoothing_function=sm.method1)
        scores_list.append(bleu)
        
    BLEU_average = sum(scores_list)/len(english_test)

    print (" BLEU average = ", BLEU_average)



# main function
if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        print("----Printing-----")
        test()
    elif sys.argv[1] == "translate":
        inp_text = input('Please input a text to translate: ')
        inp,optext,trueoptext = translate(input_text=inp_text,true_output_text=None)





