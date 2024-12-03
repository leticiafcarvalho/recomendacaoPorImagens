# **Image Similarity Search with Annoy and TensorFlow**

Este projeto utiliza a biblioteca **Annoy** para construir um índice de imagens semelhantes e o modelo **MobileNet** do TensorFlow para extrair características das imagens. O objetivo é criar um sistema de busca de similaridade de imagens com base em um vetor de características extraído de um modelo pré-treinado.

## **Funcionalidade**

O sistema permite:

- Carregar um diretório de imagens.
- Extrair características de cada imagem usando um modelo **MobileNet**.
- Construir um índice de similaridade utilizando **Annoy**.
- Realizar a busca por imagens semelhantes a partir de uma imagem de teste.

## **Tecnologias Utilizadas**

- **TensorFlow**: para carregar o modelo pré-treinado e extrair características das imagens.
- **Annoy**: para construir e pesquisar o índice de similaridade de imagens.
- **Pillow**: para o processamento e pré-processamento de imagens.
- **NumPy**: para operações de array e manipulação de dados.

## **Pré-requisitos**

Certifique-se de ter o Python 3.x e as seguintes dependências instaladas:

- **tensorflow**
- **annoy**
- **numpy**
- **Pillow**

Você pode instalar todas as dependências usando o `pip`:

```bash
pip install tensorflow annoy numpy pillow
```

## **Como Usar**

1. **Prepare o diretório de imagens**:
   - Coloque suas imagens no diretório de imagens que será lido pelo script.
   - O diretório de imagens pode ser alterado na variável `IMAGE_DIR` no código.

2. **Carregue o modelo e construa o índice**:
   - O script carrega o modelo **MobileNet** pré-treinado e extrai características das imagens.
   - Usamos **Annoy** para construir um índice de similaridade com as características extraídas.

3. **Busque por imagens semelhantes**:
   - A partir de uma imagem de teste, você pode buscar pelas imagens mais semelhantes no índice utilizando a função `find_similar_images`.

### **Exemplo de Uso**

```python
# Configuração do diretório de imagens
IMAGE_DIR = "/caminho/para/o/diretorio/de/imagens"

# Construa o índice Annoy
file_index_to_file_name = {}
index = AnnoyIndex(1280, 'angular')  # 1280 é o tamanho do vetor de características do modelo

# Listar imagens e construir o índice
for idx, file_name in enumerate(os.listdir(IMAGE_DIR)):
    file_path = os.path.join(IMAGE_DIR, file_name)
    if file_path.endswith(('.jpg', '.jpeg', '.png')):  # Verificar se é uma imagem válida
        img = preprocess_image(file_path)
        feature_vector = model(img).numpy()[0]  # Extrair características
        index.add_item(idx, feature_vector)  # Adicionar ao índice
        file_index_to_file_name[idx] = file_name

# Construir o índice com 10 árvores
index.build(10)
print("Índice Annoy criado!")

# Buscar imagens semelhantes
find_similar_images("/caminho/para/imagem/de/teste.jpg", top_k=5)
```

## **Funções**

### `preprocess_image(image_path)`
Função responsável por carregar e pré-processar a imagem para ser compatível com o modelo **MobileNet**.

### `find_similar_images(test_img_path, top_k=5)`
Função que encontra as imagens mais semelhantes à imagem de teste, usando o índice **Annoy** e as características extraídas.

### `build_index(IMAGE_DIR)`
Função que constrói o índice **Annoy** a partir de todas as imagens no diretório fornecido.


