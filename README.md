# Prova de Conceito: CHATBOT usando busca semântica com PGVECTOR

## Visão Geral

Esta prova de conceito (PoC) demonstra o potencial da busca semântica, exemplificando como ela pode ser utilizada para aprimorar a experiência do usuário em um chatbot de catálogo de produtos. Combinamos reconhecimento de intenções baseado em palavras-chave com busca vetorial para criar uma experiência conversacional inteligente, que entende consultas em linguagem natural e retorna informações relevantes sobre produtos, categorias e marcas.

O objetivo principal é validar a viabilidade técnica de usar vetores para otimizar buscas, mostrar o potencial de enriquecer o engajamento do usuário ou reduzir o tempo de busca por informações específicas em um catálogo, mesmo quando as consultas não usam termos exatos.

### Para que serve? (Benefícios e Casos de Uso)

Com a busca semântica, é possível encontrar informações não apenas por palavras-chave exatas, mas pelo significado e contexto do que o usuário procura. Permite, por exemplo, automaticamente agrupar notícias por similaridade de conteúdo, mesmo que usem palavras diferentes.

Potenciais aplicações incluem:
* **Busca Inteligente:** Ir além da busca por palavras-chave, entendendo a intenção do usuário.
* **Sistemas de Recomendação:** Sugerir itens, artigos, ou conexões mais relevantes.
* **Detecção de Anomalias/Duplicatas:** Identificar fraudes, plágio ou registros redundantes.
* **Análise de Sentimento Avançada:** Compreender nuances no feedback do cliente.
* **Chatbots e Assistentes Virtuais:** Melhorar a compreensão de perguntas e a relevância das respostas.
* **Clusterização Semântica de Documentos:** Agrupar automaticamente textos por tema.

### Como funciona (Simplificado)

De forma bem simples, o processo é o seguinte:
1.  **Transforma Informação em "Entendimento":** Dados (como textos, descrições de produtos, etc.) são convertidos em uma representação numérica especial chamada "vetor" (ou "embedding"). Cada pedaço de informação recebe uma "impressão digital" única que captura seu significado e contexto.
2.  **Armazena Essas "Impressões Digitais":** Esses vetores são guardados em um banco de dados otimizado para esse tipo de dado, como o PostgreSQL com a extensão `pgvector`.
3.  **Encontra Coisas Parecidas:** Quando uma busca é realizada ou uma recomendação é solicitada, a entrada é transformada nessa mesma "impressão digital" (vetor). O sistema então procura no banco de dados por outras "impressões digitais" que sejam matematicamente mais parecidas. O resultado são itens que se assemelham em significado, e não apenas em palavras-chave.

## Mergulho Técnico Profundo

Esta seção detalha os componentes técnicos e os conceitos por trás desta PoC.

### Arquitetura (Conceitual)

Nossa aplicação implementa uma arquitetura típica que envolve:

1. **Frontend (Cliente)** - Interface que envia solicitações para nossa API.

2. **API Flask** - Recebe consultas dos usuários através do endpoint `/api/prompt`, processa-as e retorna respostas formatadas.

3. **Serviço de Embeddings** - Utiliza o modelo `SentenceTransformer` da biblioteca `sentence-transformers` para converter textos em vetores.

4. **PostgreSQL com pgvector** - Armazena e consulta os dados do catálogo, incluindo os vetores dos produtos e categorias para busca por similaridade.

5. **Sistema de Reconhecimento de Intenções** - Identifica o objetivo da consulta do usuário com base em palavras-chave normalizadas e armazenadas em cache.

### Tecnologia Central: `pgvector`

O coração desta PoC é o **`pgvector`**, uma extensão para o PostgreSQL que adiciona suporte para armazenamento e busca de vetores de alta dimensionalidade. Isso nos permite realizar buscas por similaridade diretamente no banco de dados relacional.

* **O que são Vetores (Embeddings)?**
    No contexto de IA e Machine Learning, um "embedding" é uma representação vetorial (uma lista de números) de um dado (texto, imagem, áudio, etc.). Esses vetores são gerados por modelos de aprendizado profundo de forma que itens semanticamente similares tenham vetores próximos no espaço vetorial. Por exemplo, as frases "Como está o tempo hoje?" e "Qual a previsão meteorológica para agora?" terão vetores muito mais próximos entre si do que da frase "Qual a capital da França?". Os vetores podem ser gerados usando modelos como os da família `sentence-transformers` ou APIs de embeddings como as da OpenAI. **Neste projeto, utilizamos especificamente o modelo `neuralmind/bert-base-portuguese-cased`, otimizado para a língua portuguesa, que gera embeddings de 768 dimensões.**

* **Por que `pgvector`?**
    * **Integração com Dados Relacionais:** Permite combinar buscas vetoriais com filtros SQL tradicionais em outras colunas da tabela (ex: buscar produtos similares *que estão em estoque* e *dentro de uma faixa de preço*).
    * **Transações ACID:** Beneficia-se da robustez, consistência, isolamento e durabilidade do PostgreSQL.
    * **Ecossistema Maduro:** Aproveita as ferramentas, backups, replicação, e a vasta comunidade do PostgreSQL.
    * **Simplicidade Operacional:** Para muitos casos de uso, evita a necessidade de manter e operar um sistema de busca vetorial separado, reduzindo a complexidade da stack tecnológica.

* **Armazenamento de Vetores:**
    Com `pgvector`, criamos colunas do tipo `vector(N)`, onde `N` é a dimensionalidade do vetor. Por exemplo, se um modelo de embedding gera vetores de 384 dimensões, a coluna seria `embedding_column vector(384)`.

### Busca por Similaridade

A busca por similaridade ocorre ao comparar esses vetores. A similaridade entre dois vetores é medida por uma função de distância ou similaridade:

* **Distância Euclidiana (L2):** Operador `vector <-> vector`
    * Mede a distância "reta" entre dois pontos no espaço vetorial. Quanto menor a distância, mais similares são os itens.
    * Fórmula: $\sqrt{\sum{(v1_i - v2_i)^2}}$

* **Similaridade de Cosseno:** Calculada como `1 - (vector <=> vector)`
    * Mede o cosseno do ângulo entre dois vetores. Varia de -1 (exatamente opostos) a 1 (exatamente iguais). O operador `<=>` no `pgvector` retorna a *distância* de cosseno (1 - similaridade de cosseno). Um valor mais próximo de 1 na similaridade de cosseno (ou 0 na distância de cosseno) indica maior semelhança.
    * Fórmula (Similaridade): $\frac{(V1 \cdot V2)}{(\|V1\| \cdot \|V2\|)}$
    * Ideal para: Dados textuais e outros casos onde a orientação dos vetores (direção) é mais importante que a magnitude. É a métrica mais comum para embeddings de texto.

* **Produto Interno (Inner Product):** Operador `vector <#> vector`
    * Retorna o negativo do produto interno. Para usar como medida de similaridade (onde maior é melhor), pode-se usar `- (embedding <#> query_vector)`.
    * Fórmula (Produto Interno): $\sum{(v1_i \cdot v2_i)}`

    * Se os vetores são normalizados (comprimento unitário), o produto interno é equivalente à similaridade de cosseno em termos de ordenação.

A métrica de similaridade escolhida frequentemente é a Similaridade de Cosseno (usando o operador `<=>`) para embeddings de texto, pois a direção dos vetores é um indicador relevante de similaridade semântica.

Uma consulta típica envolve:
1.  Converter o input da busca (ex: uma frase) em um vetor usando o mesmo modelo de embedding.
2.  Usar este vetor na cláusula `ORDER BY` de uma consulta SQL para encontrar os vetores mais próximos na tabela.

### Indexação em `pgvector` (ANN - Approximate Nearest Neighbor)

Para datasets grandes, calcular a distância exata para todos os vetores (busca "k-NN exato") pode ser lento. `pgvector` suporta índices para busca de vizinhos mais próximos aproximados (ANN), que sacrificam um pouco de precisão (recall) por uma velocidade de busca muito maior.

* **IVFFlat (`ivfflat`):**
    * **Como funciona:** Divide o dataset em `lists` partições (clusters) usando k-means. Na busca, apenas um subconjunto `probes` dessas partições (as mais próximas do vetor de consulta) é inspecionado.
    * **Trade-offs:** Bom equilíbrio entre velocidade de construção do índice e velocidade de busca/acurácia. Requer reconstrução do índice se os dados mudarem significativamente.

* **HNSW (`hnsw`):**
    * **Como funciona:** Constrói um grafo multinível onde os nós são os vetores e as arestas conectam vizinhos próximos. A busca navega por este grafo de forma eficiente.
    * **Trade-offs:** Geralmente oferece melhor acurácia/velocidade que IVFFlat, especialmente para datasets maiores ou mais complexos. A construção do índice pode ser mais lenta. Adiciona novos dados de forma incremental.

A utilização de um índice ANN é crucial para otimizar consultas em datasets grandes, embora para datasets de demonstração muito pequenos, possa não ser estritamente necessário. A escolha do tipo de índice e seus parâmetros (como `lists` e `probes` para IVFFlat, ou `m`, `ef_construction`, e `ef_search` para HNSW) depende do tamanho do dataset, da dimensionalidade dos vetores e do equilíbrio desejado entre velocidade e precisão.

### "Regras" e Lógica de Negócio Aplicada

O termo "regras" aqui pode abranger várias lógicas aplicadas em conjunto com a busca vetorial:

* **Filtragem Híbrida (Metadados + Vetorial):** A grande força do `pgvector` é combinar a busca por similaridade com filtros SQL tradicionais. Isso permite, por exemplo, encontrar "itens similares AO ITEM X, MAS APENAS aqueles da CATEGORIA Y e COM PREÇO ABAIXO DE Z".

* **Limiares de Similaridade (Thresholds):** Definir um score mínimo de similaridade para que um item seja considerado relevante. Resultados abaixo desse limiar são descartados.

* **Re-ranking:** Após obter os N resultados mais similares, podem-se aplicar regras adicionais para reordená-los. Por exemplo, dar um "boost" em itens mais recentes, mais populares, ou com melhor avaliação.

* **Pré-processamento de Dados e Consultas:** Regras sobre como os dados são limpos, normalizados (ex: minúsculas, remoção de pontuação) e transformados em vetores. É crucial que o texto da consulta passe pelo mesmo pré-processamento que os textos armazenados antes de serem vetorizados.

Estas "regras" são aplicadas para refinar os resultados da busca vetorial, tornando-os mais alinhados com os requisitos específicos do negócio.

### Fluxo de Dados Detalhado (Conceitual)

1.  **Ingestão de Dados (se aplicável):**
    a.  Dados brutos são lidos.
    b.  Os campos textuais relevantes passam por um pré-processamento (ex: limpeza, normalização).
    c.  O texto pré-processado é enviado para um modelo de embedding para gerar um vetor.
    d.  Os dados originais, junto com o vetor gerado, são inseridos na tabela apropriada do PostgreSQL.

2.  **Processamento de Consulta:**
    a.  A consulta do usuário é recebida.
    b.  A consulta passa pelo mesmo pré-processamento da etapa de ingestão.
    c.  A consulta pré-processada é vetorizada usando o mesmo modelo de embedding, gerando um `query_vector`.
    d.  Uma consulta SQL é construída para buscar na tabela, utilizando o `query_vector` para ordenação por similaridade e, opcionalmente, aplicando filtros de metadados.

3.  **Pós-processamento e Apresentação:**
    a.  Os resultados mais similares são retornados.
    b.  (Opcional) Pode-se aplicar um threshold de similaridade e/ou re-ranking.
    c.  Os resultados finais são formatados e apresentados ao usuário.

## Implementação do Chatbot

A implementação desta prova de conceito contém um chatbot baseado em Flask que responde consultas sobre um catálogo de produtos, utilizando uma combinação de:

1. **Reconhecimento de intenções baseado em palavras-chave**
2. **Busca semântica com pgvector**
3. **Consultas SQL convencionais**

### Modelo de Embedding

Utilizamos o modelo `neuralmind/bert-base-portuguese-cased`, específico para textos em português, que gera vetores de 768 dimensões. A função `get_embedding` transforma o texto do usuário em vetores, aplicando normalização (conversão para minúsculas e remoção de espaços extras).

### Banco de Dados e Estrutura

O sistema utiliza PostgreSQL com a extensão pgvector, com tabelas principais:

- `secoes_catalogo`: Categorias e subcategorias do catálogo
- `itens_secao`: Produtos com suas informações, incluindo embeddings
- `marcas`: Marcas dos produtos
- `palavras_chave_intencao`: Palavras-chave para reconhecer intenções

### Reconhecimento de Intenções

O chatbot identifica 7 tipos de intenções principais:

1. **Catálogo Geral** - Responde com as categorias principais do catálogo
   - *Exemplo*: "Mostre o catálogo" ou "O que vocês vendem?"

2. **Produtos por Tipo e Marca** - Busca produtos filtrando por tipo e marca
   - *Exemplo*: "Quero ver notebook da Dell"

3. **Produtos por Categoria** - Lista produtos ou subcategorias de uma seção
   - *Exemplo*: "O que tem na seção de eletrônicos?" ou "Mostre computadores"

4. **Produtos por Marca** - Lista produtos de uma marca específica
   - *Exemplo*: "Produtos da Samsung" ou "O que tem da Apple?"

5. **Produtos por Tipo** - Busca produtos por descrição ou nome genérico
   - *Exemplo*: "Tem celular?" ou "Cafeteiras"

6. **Marcas por Categoria** - Lista marcas disponíveis em uma categoria
   - *Exemplo*: "Quais marcas de televisores vocês têm?"

7. **Fallback (Busca Semântica)** - Quando nenhuma intenção é claramente identificada, o sistema recorre à busca vetorial para encontrar produtos ou categorias semanticamente similares
   - *Exemplo*: Qualquer consulta que não se encaixe nas intenções anteriores

### Processamento de Consultas

As consultas do usuário seguem este fluxo:
1. **Pré-processamento**: Remoção de acentos, conversão para minúsculas
2. **Tentativa de reconhecer intenção**: Análise de prefixos e padrões de texto
3. **Execução de consultas específicas**: Conforme a intenção identificada
4. **Fallback**: Se nenhuma intenção for identificada, o sistema usa similaridade vetorial para encontrar o produto ou categoria mais próximo da consulta

### Limiares de Similaridade

Para a busca semântica (fallback), o sistema utiliza limiares de similaridade:
- Para produtos: distância vetorial < 7.5
- Para categorias: distância vetorial < 8.5

Produtos têm prioridade sobre categorias quando a diferença de distância é significativa (> 0.2).

### Características Adicionais

- **Cache de palavras-chave**: O sistema carrega e normaliza as palavras-chave de intenção na inicialização
- **Hierarquia de categorias**: Suporte a categorias e subcategorias aninhadas
- **Formatação de resultados**: As respostas são formatadas de maneira amigável, incluindo preços e datas

### Exemplos de Consultas e Respostas

#### Exemplo 1: Catálogo Geral
**Consulta:**
```json
POST /api/prompt
{
  "prompt": "O que vocês vendem?"
}
```

**Resposta:**
```json
{
  "message": "Nosso catálogo principal inclui: Eletrônicos, Eletrodomésticos, Móveis, Decoração, Jardinagem. Pergunte 'o que tem em [nome da seção]?' para detalhes."
}
```

#### Exemplo 2: Produtos por Tipo e Marca
**Consulta:**
```json
POST /api/prompt
{
  "prompt": "Quero ver notebook da Dell"
}
```

**Resposta:**
```json
{
  "message": "Para 'notebook' da marca 'Dell': - Dell XPS 13 (Marca: Dell) Preço: R$8999.90 Validade: N/A Desc: Notebook premium ultrafino com Intel Core i7 e 16GB RAM... - Dell Inspiron 15 (Marca: Dell) Preço: R$4599.00 Validade: N/A Desc: Notebook para uso diário com Intel Core i5 e 8GB RAM..."
}
```

#### Exemplo 3: Fallback - Busca Semântica
**Consulta:**
```json
POST /api/prompt
{
  "prompt": "algo para guardar bebidas geladas"
}
```

**Resposta:**
```json
{
  "message": "Encontrei 'Refrigerador Side by Side'. Desc: Refrigerador com dispensador de água e gelo na porta. Marca: Samsung. Preço: R$7499.00. Validade: N/A."
}

