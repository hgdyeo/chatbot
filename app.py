from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from flask import Response
from chatbot import *
from santa import *
import pickle
from flask import Response, Flask, render_template, request
import os



MAX_LENGTH = 10

app = Flask(__name__)
base_dir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(base_dir, 'kubrick.db')

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token



@app.route('/santa')
def santa():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat')
def my_form():
    return render_template('main.html')


@app.route('/chat', methods=['post'])
def get_consultant():
    sentence = request.form['text']

    response = evaluateInput(voc, encoder, decoder, sentence)

    return render_template('main.html', input=sentence, output=response)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('models/voc.pickle', 'rb') as f:
        voc = pickle.load(f)

    attn_model = 'dot'

    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    loadFilename = 'models/4000_checkpoint.tar'
    checkpoint_iter = 4000

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename, map_location=device)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    encoder.eval()
    decoder.eval()

    app.run(debug=True, port=5000, host='0.0.0.0')
