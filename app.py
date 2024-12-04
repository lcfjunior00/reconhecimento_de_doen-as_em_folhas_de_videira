import streamlit as st
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    interpreter = tf.lite.Interpreter(model_path='models/modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    upload_file = st.file_uploader('Arraste e solte sua imagem', type=['png', 'jpg', 'jpeg'])

    if upload_file is not None:
        image_data = upload_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Image carregada com sucesso')

        input_shape = input_details[0]['shape']  # [1, altura, largura, canais]
        target_height, target_width = input_shape[1], input_shape[2]

        image = image.resize((target_width, target_height))
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title = 'Probabilidade de classes de Doenças em Uvas.')
    
    st.plotly_chart(fig)



def main():

    st.set_page_config(
        page_title = 'Classifica folhas de Videira'
    )

    st.write('# Classifica folhas de Videira')

    #Carrega modelo:
    interpreter = carrega_modelo()

    #Carrega imagem:
    image = carrega_imagem()
    
    #Classifica:
    if image is not None:

        previsao(interpreter, image)

    return

if __name__ == '__main__':
    main()