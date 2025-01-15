# Importing essential libraries
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras as kr
import tensorflow as tf
from keras import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input
from tensorflow.keras.models import Sequential


class Patches (Layer):
  def __init__(self, patch_size):
    super(Patches, self).__init__()
    self.patch_size = patch_size

  def call(self, images):
    batch_size = tf.shape(images)[0]

    patches = tf.image.extract_patches(
      images=images,
      sizes=[1, self.patch_size, self.patch_size, 1],
      strides=[1, self.patch_size, self.patch_size, 1],
      rates=[1, 1, 1, 1],
      padding='VALID'
    )

    dim = patches.shape[-1]

    patches = tf.reshape(patches, (batch_size, -1, dim))

    return patches
  
class PatchEmbedding(Layer):
  def __init__(self, patch_size, image_size, projection_dim):
    super(PatchEmbedding, self).__init__()

    self.num_patches = (image_size // patch_size) **2

    self.cls_token = self.add_weight(
      name="cls_token",
      shape=[1, 1, projection_dim],
      initializer= kr.initializers.RandomNormal(),
      dtype= tf.float32
    )

    self.patches = Patches(patch_size)

    self.projection = Dense(units = projection_dim)

    self.position_embdding = self.add_weight(
      name="position_embeddings",
      shape=[self.num_patches +1, projection_dim],
      initializer= kr.initializers.RandomNormal(),
      dtype=tf.float32
    )

  def call(self, images):
    patch= self.patches(images)
    encoded_patches = self.projection(patch)
    batch_size =  tf.shape(images)[0]
    hidden_size = tf.shape(encoded_patches)[-1]
    cls_broadcasted = tf.cast(
      tf.broadcast_to(self.cls_token, [batch_size, 1, hidden_size]),
      dtype=images.dtype
    )
    encoded_patches = tf.concat([cls_broadcasted, encoded_patches], axis=1)
    encoded_patches += self.position_embdding
    return encoded_patches

class MLPBlock(Layer):
  def __init__(self, hidden_layers, dropout = 0.1, activation= 'gelu'):
    super(MLPBlock, self).__init__()

    layers= []
    for num_units in hidden_layers:
      layers.extend([
        Dense(num_units, activation= activation),
        Dropout(dropout)
      ])

    self.mlp = Sequential(layers)

  def call(self, inputs):
    outputs= self.mlp(inputs)
    return outputs

class TransformerBlock(Layer):
  def __init__(self, num_heads, D, hidden_layers, dropout= 0.1, norm_eps= 1e-12):
    super(TransformerBlock, self).__init__()
    self.norm = LayerNormalization(epsilon=norm_eps)
    self.attention = MultiHeadAttention(
      num_heads=num_heads, key_dim=D // num_heads, dropout=dropout
    )
    self.mlp= MLPBlock(hidden_layers, dropout)

  def call(self, inputs):
    norm_attention = self.norm(inputs)
    attention = self.attention(query= norm_attention, value = norm_attention)
    attention += inputs
    outputs= self.mlp(self.norm(attention))
    outputs += attention
    return outputs
  
class TransformerEncoder(Layer):
  def __init__(self, num_layers, num_heads, D, mlp_dim, dropout= 0.1, norm_eps= 1e-12):
    super(TransformerEncoder, self).__init__()

    transformer_blocks = []
    for _ in range(num_layers):
      block = TransformerBlock(
          num_heads=num_heads,
          D= D,
          hidden_layers= [mlp_dim, D],
          dropout= dropout,
          norm_eps= norm_eps
      )

      transformer_blocks.append(block)
    self.encoder = Sequential(transformer_blocks)

  def call(self, inputs):
    outputs = self.encoder(inputs)
    return outputs 

class ViT(Model):
    def __init__(self, num_classes, num_layers= 6, num_heads= 8, D=128, mlp_dim= 640, patch_size= 4, image_size= 28, dropout= 0.2, norm_eps= 1e-12):
        super(ViT, self).__init__()
        
        self.embedding= PatchEmbedding(patch_size, image_size, D)
        
        self.encoder = TransformerEncoder(
          num_layers=num_layers,
          num_heads=num_heads,
          D=D,
          mlp_dim=mlp_dim,
          dropout=dropout,
          norm_eps=norm_eps
        )
        
        self.mlp_head = Sequential([
          LayerNormalization(epsilon= norm_eps),
          Dense(mlp_dim),
          Dropout(dropout),
          Dense(num_classes, activation='sigmoid')
        ])
        
        self.last_layer_norm= LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        # total_paramters = 0
        embedded = self.embedding(inputs)
        encoded = self.encoder(embedded)
        embedded_cls = encoded[:, 0]
        y = self.last_layer_norm(embedded_cls)
        output = self.mlp_head(y)
        return output



img_size= 28
channel = 1

inputs = Input(shape=(img_size, img_size, channel)) 
vit_model = ViT(
      num_classes=1,
      image_size= img_size
  )

outputs = vit_model(inputs)  

model = Model(inputs=inputs, outputs=outputs)

# optimizer = kr.optimizers.AdamW(
#     learning_rate=0.001, weight_decay=1e-4
#   )

# best hyper
optimizer = kr.optimizers.AdamW(
    learning_rate=0.0002957, weight_decay=1.9091e-06
  )

model.compile(
    optimizer=optimizer, 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

model.summary()

model.load_weights('vit_pneumonia_weights.best.weights.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']  
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((28, 28))
        img = img.convert('L')
        img_array = np.array(img)
        img_array = img_array / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        # Dự đoán với mô hình
        percent = model.predict(img_array)
        percent = percent[0][0]
        predictions = (percent >= 0.5).astype(int)

        if(predictions == 1):
            predictions = f"Bạn đã bị viêm phổi, tỉ lệ bệnh là: {percent * 100:.2f}%"
        else :
            predictions = f"Bạn không bị viêm phổi, tỉ lệ bệnh là: {percent * 100:.2f}%"
         
        # Trả kết quả dự đoán
        return render_template('index.html', prediction=predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        

if __name__ == '__main__':
    app.run(debug=True)
