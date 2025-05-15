# Prova de Conceito: CHATBOT usando busca semantica com PGVECTOR

## Visão Geral (Nível de Negócio)

Esta prova de conceito (PoC) demonstra [**o quê? Ex: o poder da busca semântica para melhorar a experiência do usuário em nossa plataforma de e-commerce / como podemos identificar documentos similares em grandes volumes de texto / um sistema de recomendação inteligente baseado em embeddings vetoriais**].

O objetivo principal é [**qual? Ex: validar a viabilidade técnica de usar vetores para X / mostrar o potencial de Y para aumentar o engajamento / reduzir o tempo de busca por Z**].

### Para que serve? (Benefícios e Casos de Uso)

Imagine poder [**cenário de benefício 1, ex: encontrar produtos não apenas por palavras-chave exatas, mas pelo significado e contexto do que o cliente procura**]. Ou então, [**cenário de benefício 2, ex: automaticamente agrupar notícias por similaridade de conteúdo, mesmo que usem palavras diferentes**].

Potenciais aplicações incluem:
*   **Busca Inteligente:** Ir além da busca por palavras-chave, entendendo a intenção do usuário.
*   **Sistemas de Recomendação:** Sugerir itens, artigos, ou conexões mais relevantes.
*   **Detecção de Anomalias/Duplicatas:** Identificar fraudes, plágio ou registros redundantes.
*   **Análise de Sentimento Avançada:** Compreender nuances no feedback do cliente.
*   **Chatbots e Assistentes Virtuais:** Melhorar a compreensão de perguntas e a relevância das respostas.
*   **Clusterização Semântica de Documentos:** Agrupar automaticamente textos por tema.

### Como funciona (Simplificado para Negócios)

De forma bem simples, esta PoC faz o seguinte:
1.  **Transforma Informação em "Entendimento":** Pegamos dados (como textos, descrições de produtos, etc.) e os convertemos em uma representação numérica especial chamada "vetor" (ou "embedding"). Pense nisso como dar a cada pedaço de informação uma "impressão digital" única que captura seu significado e contexto.
2.  **Armazena Essas "Impressões Digitais":** Guardamos esses vetores em um banco de dados otimizado para esse tipo de dado (neste caso, usamos o PostgreSQL com a extensão `pgvector`).
3.  **Encontra Coisas Parecidas:** Quando você faz uma busca (ex: "celular bom para fotos noturnas") ou quer uma recomendação, transformamos sua pergunta/item de referência nessa mesma "impressão digital" (vetor) e procuramos no banco de dados por outras "impressões digitais" que sejam matematicamente mais parecidas. O resultado são itens que se assemelham em significado, e não apenas em palavras-chave.

## Primeiros Passos (Para Desenvolvedores)

### Pré-requisitos
*   **PostgreSQL:** Versão [**X.Y ou superior**] (ex: 13+).
*   **Extensão `pgvector`:** Instalada e habilitada no seu banco de dados PostgreSQL. (Veja https://github.com/pgvector/pgvector)
*   **[Linguagem de Programação]:** Ex: Python 3.8+, Node.js 16+, Java 11+, etc.
*   **Bibliotecas Adicionais:** Ex: `psycopg2-binary` (para Python), `node-postgres` (para Node.js), `sentence-transformers` (para gerar embeddings em Python), etc. [**Liste as dependências principais aqui**].
*   **[Outras Ferramentas]:** Ex: Docker (se aplicável para rodar o PostgreSQL), um modelo de embedding (ex: baixado do Hugging Face ou via API).

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DO_DIRETORIO_DA_POC]
    ```
2.  **Configure o Banco de Dados:**
    *   Certifique-se de que o PostgreSQL está rodando.
    *   Crie um banco de dados para a PoC: `CREATE DATABASE nome_do_banco_poc;`
    *   Conecte-se ao banco (`psql -d nome_do_banco_poc`) e habilite a extensão `pgvector`:
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        ```
    *   (Se houver) Rode os scripts de criação de tabelas:
        ```bash
        psql -d nome_do_banco_poc -f /path/to/your/schema.sql
        ```
        Ou execute os comandos SQL para criar sua tabela com uma coluna do tipo `vector(DIMENSOES)`, ex: `CREATE TABLE items (id SERIAL PRIMARY KEY, content TEXT, embedding vector(384));`

3.  **Instale as dependências do projeto:**
    *(Exemplo para Python com `requirements.txt`)*
    ```bash
    pip install -r /path/to/your/requirements.txt
    ```
    *(Exemplo para Node.js com `package.json`)*
    ```bash
    npm install
    ```

4.  **Configure as variáveis de ambiente:**
    Crie um arquivo `.env` (ou similar) baseado no `.env.example` (se houver) e preencha com suas credenciais de banco de dados, caminhos para modelos, chaves de API (se usar embeddings de serviços pagos), etc.
    Exemplo de `.env`:
    ```env
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=nome_do_banco_poc
    DB_USER=seu_usuario
    DB_PASSWORD=sua_senha
    # EMBEDDING_MODEL_PATH=/path/to/model (se local)
    ```

### Como Rodar a PoC

1.  **Ingestão de Dados (se necessário):**
    Se sua PoC envolve popular o banco com vetores primeiro:
    ```bash
    python /path/to/your/ingest_data.py --input /path/to/your/dataset.csv
    ```
2.  **Executar a Busca/Funcionalidade Principal:**
    ```bash
    python /path/to/your/main_script.py --query "qual o melhor celular para fotos?"
    ```
    [**Adapte com os comandos reais para rodar sua PoC**]

### Exemplo de Uso

*(Mostre um exemplo prático de entrada e a saída esperada, ou como interagir com a PoC).*

**Entrada:**
Uma consulta de texto, como: "documentos sobre inteligência artificial aplicada à saúde".

**Saída Esperada:**
Uma lista de documentos ou itens do seu banco de dados que são semanticamente similares à consulta, ordenados por relevância.
```
Resultado 1: "Título do Artigo sobre IA em Diagnósticos Médicos" (Similaridade: 0.89)
Resultado 2: "Paper Científico: Machine Learning para Previsão de Doenças" (Similaridade: 0.85)
...
```

## Mergulho Técnico Profundo

Esta seção detalha os componentes técnicos e os conceitos por trás desta PoC.

### Arquitetura (Opcional)

*[Se relevante, descreva brevemente os principais componentes e como eles interagem. Um diagrama simples pode ser útil aqui. Ex: Cliente -> API -> Serviço de Embedding -> pgvector -> Resultados]*

### Tecnologia Central: `pgvector`

O coração desta PoC é o **`pgvector`**, uma extensão para o PostgreSQL que adiciona suporte para armazenamento e busca de vetores de alta dimensionalidade. Isso nos permite realizar buscas por similaridade diretamente no banco de dados relacional.

*   **O que são Vetores (Embeddings)?**
    No contexto de IA e Machine Learning, um "embedding" é uma representação vetorial (uma lista de números) de um dado (texto, imagem, áudio, etc.). Esses vetores são gerados por modelos de aprendizado profundo (como Transformers, redes neurais convolucionais, etc.) de forma que itens semanticamente similares tenham vetores próximos no espaço vetorial.
    Por exemplo, as frases "Como está o tempo hoje?" e "Qual a previsão meteorológica para agora?" terão vetores muito mais próximos entre si do que da frase "Qual a capital da França?".
    Nesta PoC, os vetores são gerados usando [**EXPLIQUE QUAL MODELO/TÉCNICA VOCÊ USOU AQUI**, ex: o modelo `sentence-transformers/all-MiniLM-L6-v2` (que gera vetores de 384 dimensões), ou a API de embeddings da OpenAI (ex: `text-embedding-ada-002` com 1536 dimensões), ou um modelo treinado internamente].

*   **Por que `pgvector`?**
    *   **Integração com Dados Relacionais:** Permite combinar buscas vetoriais com filtros SQL tradicionais em outras colunas da sua tabela (ex: buscar produtos similares *que estão em estoque* e *dentro de uma faixa de preço*). Isso é uma grande vantagem sobre bancos de vetores dedicados que podem exigir sincronização de dados.
    *   **Transações ACID:** Beneficia-se da robustez, consistência, isolamento e durabilidade do PostgreSQL.
    *   **Ecossistema Maduro:** Aproveita as ferramentas, backups, replicação, e a vasta comunidade do PostgreSQL.
    *   **Simplicidade Operacional:** Para muitos casos de uso, evita a necessidade de manter e operar um sistema de busca vetorial separado, reduzindo a complexidade da stack tecnológica.

*   **Armazenamento de Vetores:**
    Com `pgvector`, criamos colunas do tipo `vector(N)`, onde `N` é a dimensionalidade do vetor. Por exemplo, se um modelo de embedding gera vetores de 384 dimensões, a coluna seria `embedding_column vector(384)`.

### Busca por Similaridade

A "mágica" acontece ao comparar esses vetores. A similaridade entre dois vetores é medida por uma função de distância ou similaridade:

*   **Distância Euclidiana (L2):** `vector <-> vector`
    *   Mede a distância "reta" entre dois pontos no espaço vetorial. Quanto menor a distância, mais similares são os itens.
    *   Fórmula: `sqrt(sum((v1_i - v2_i)^2))`
    *   Uso: `SELECT id FROM items ORDER BY embedding <-> query_vector LIMIT 5;`

*   **Similaridade de Cosseno:** `1 - (vector <=> vector)`
    *   Mede o cosseno do ângulo entre dois vetores. Varia de -1 (exatamente opostos) a 1 (exatamente iguais). O operador `<=>` no `pgvector` retorna a *distância* de cosseno (1 - similaridade de cosseno), então para obter a similaridade (onde maior é melhor), você usa `1 - (embedding <=> query_vector)`. Um valor mais próximo de 1 na similaridade de cosseno indica maior semelhança.
    *   Fórmula (Similaridade): `(V1 · V2) / (||V1|| * ||V2||)`
    *   Uso (ordenando por distância, menor é melhor): `SELECT id FROM items ORDER BY embedding <=> query_vector LIMIT 5;`
    *   Ideal para: Dados textuais e outros casos onde a orientação dos vetores (direção) é mais importante que a magnitude. É a métrica mais comum para embeddings de texto.

*   **Produto Interno (Inner Product):** `vector <#> vector`
    *   Retorna o negativo do produto interno. Para similaridade (onde maior é melhor), você pode usar `- (embedding <#> query_vector)`.
    *   Fórmula (Produto Interno): `sum(v1_i * v2_i)`
    *   Uso (ordenando por produto interno negativo, menor é melhor): `SELECT id FROM items ORDER BY embedding <#> query_vector LIMIT 5;`
    *   Se os vetores são normalizados (comprimento unitário), o produto interno é equivalente à similaridade de cosseno em termos de ordenação.

Nesta PoC, a métrica de similaridade utilizada é [**ESPECIFIQUE A MÉTRICA E O PORQUÊ**, ex: Similaridade de Cosseno (usando o operador `<=>`), pois estamos lidando com embeddings de texto e a direção dos vetores é o indicador mais relevante de similaridade semântica].

Uma consulta típica envolve:
1.  Converter o input da busca (ex: uma frase) em um vetor usando o mesmo modelo de embedding.
2.  Usar este vetor na cláusula `ORDER BY` de uma consulta SQL para encontrar os vetores mais próximos na tabela.
    ```sql
    -- Exemplo com distância de cosseno
    SELECT id, content, 1 - (embedding <=> '[...vetor da consulta como string...]') AS similarity
    FROM items
    ORDER BY embedding <=> '[...vetor da consulta como string...]'
    LIMIT 10;
    ```

### Indexação em `pgvector` (ANN - Approximate Nearest Neighbor)

Para datasets grandes, calcular a distância exata para todos os vetores (busca exaustiva ou "k-NN exato") pode ser lento. `pgvector` suporta índices para busca de vizinhos mais próximos aproximados (ANN), que sacrificam um pouco de precisão (recall) por uma velocidade de busca muito maior.

*   **IVFFlat (`ivfflat`):**
    *   **Como funciona:** Divide o dataset em `lists` partições (clusters) usando k-means. Na busca, apenas um subconjunto `probes` dessas partições (as mais próximas do vetor de consulta) é inspecionado.
    *   **Criação:** `CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);` (Ajuste `lists` e o `vector_operator_ops` conforme a métrica: `vector_l2_ops`, `vector_cosine_ops`, `vector_ip_ops`).
    *   **Parâmetros de busca:** `SET ivfflat.probes = 10;` (Ajuste `probes` antes da consulta).
    *   **Trade-offs:** Bom equilíbrio entre velocidade de construção do índice e velocidade de busca/acurácia. Requer reconstrução do índice se os dados mudarem significativamente. `lists` geralmente é `N/1000` para N < 1M, ou `sqrt(N)` para N > 1M. `probes` é um valor menor, como `sqrt(lists)`.

*   **HNSW (`hnsw`):**
    *   **Como funciona:** Constrói um grafo multinível onde os nós são os vetores e as arestas conectam vizinhos próximos. A busca navega por este grafo de forma eficiente.
    *   **Criação:** `CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);` (Ajuste `m` e `ef_construction`).
    *   **Parâmetros de busca:** `SET hnsw.ef_search = 40;` (Ajuste `ef_search` antes da consulta).
    *   **Trade-offs:** Geralmente oferece melhor acurácia/velocidade que IVFFlat, especialmente para datasets maiores ou mais complexos. A construção do índice pode ser mais lenta. Adiciona novos dados de forma incremental.

Nesta PoC, [**ESPECIFIQUE SE USA ÍNDICE, QUAL TIPO, E PORQUÊ, OU SE NÃO USA E PORQUÊ** - ex: "utilizamos um índice IVFFlat com 100 listas e `vector_cosine_ops` para otimizar as consultas em nosso dataset de 10.000 itens. O parâmetro `probes` é definido como 10 em tempo de execução." ou "não utilizamos indexação ANN nesta PoC devido ao tamanho reduzido do dataset de demonstração (X itens), mas seria um passo crucial para produção."].

### "Regras" e Lógica de Negócio Aplicada

O termo "regras" aqui pode abranger várias lógicas aplicadas em conjunto com a busca vetorial:

*   **Filtragem Híbrida (Metadados + Vetorial):** A grande força do `pgvector` é combinar a busca por similaridade com filtros SQL tradicionais.
    ```sql
    SELECT id, content, 1 - (embedding <=> '[...vetor...]') AS similarity
    FROM items
    WHERE category = 'eletronicos' AND price < 500.00 AND in_stock = TRUE -- Filtros de metadados
    ORDER BY embedding <=> '[...vetor...]' -- Ordenação por similaridade
    LIMIT 10;
    ```
    Isso permite, por exemplo, encontrar "produtos similares AO PRODUTO X, MAS APENAS aqueles da MARCA Y e COM PREÇO ABAIXO DE Z".

*   **Limiares de Similaridade (Thresholds):** Definir um score mínimo de similaridade para que um item seja considerado relevante. Resultados abaixo desse limiar são descartados, mesmo que estejam entre os "mais próximos".
    ```sql
    SELECT id, content, similarity
    FROM (
        SELECT id, content, 1 - (embedding <=> '[...vetor...]') AS similarity
        FROM items
        WHERE category = 'noticias' -- Exemplo de filtro
        ORDER BY similarity DESC
        LIMIT 50 -- Pega um conjunto maior para depois filtrar pelo threshold
    ) AS ranked_results
    WHERE similarity > 0.75; -- Threshold de similaridade
    ```

*   **Re-ranking:** Após obter os N resultados mais similares, podemos aplicar regras adicionais para reordená-los. Por exemplo, dar um "boost" em itens mais recentes, mais populares, ou com melhor avaliação, mesmo que a similaridade vetorial seja ligeiramente menor.

*   **Pré-processamento de Dados e Consultas:** Regras sobre como os dados são limpos, normalizados (ex: minúsculas, remoção de pontuação) e transformados em vetores. É crucial que o texto da consulta passe pelo mesmo pré-processamento que os textos armazenados antes de serem vetorizados.

Nesta PoC, as seguintes "regras" ou lógicas são aplicadas: [**DESCREVA AS REGRAS ESPECÍFICAS DA SUA POC AQUI. Ex: "Filtramos por categoria antes da busca vetorial e aplicamos um threshold de similaridade de 0.7."**].

### Fluxo de Dados Detalhado (Exemplo)

1.  **Ingestão de Dados (se aplicável):**
    a.  Os dados brutos (ex: um CSV com `id`, `descricao_produto`, `categoria`, `preco`) são lidos.
    b.  A `descricao_produto` passa por um pré-processamento (ex: limpeza de HTML, minúsculas).
    c.  A `descricao_produto` pré-processada é enviada para o modelo de embedding (ex: `all-MiniLM-L6-v2`) para gerar um vetor de 384 dimensões.
    d.  O `id`, `descricao_produto` original, `categoria`, `preco` e o `embedding` gerado são inseridos na tabela `products` do PostgreSQL.

2.  **Processamento de Consulta:**
    a.  A consulta do usuário (ex: "notebook leve para trabalho") é recebida.
    b.  A consulta passa pelo mesmo pré-processamento da etapa 1.b.
    c.  A consulta pré-processada é vetorizada usando o mesmo modelo de embedding, gerando um `query_vector`.
    d.  Uma consulta SQL é construída para buscar na tabela `products`:
        ```sql
        SELECT id, descricao_produto, preco, 1 - (embedding <=> :query_vector) as similarity
        FROM products
        WHERE categoria = 'informatica' -- Filtro opcional
        ORDER BY embedding <=> :query_vector
        LIMIT 10;
        ```
        O `:query_vector` é substituído pelo vetor gerado.

3.  **Pós-processamento e Apresentação:**
    a.  Os resultados (top 10 produtos mais similares da categoria 'informatica') são retornados.
    b.  (Opcional) Pode-se aplicar um threshold de similaridade para remover resultados menos relevantes.
    c.  Os resultados finais são formatados e apresentados ao usuário.

### Pontos Chave no Código

*   **Geração de Embeddings:** Veja o arquivo/módulo `[caminho/absoluto/para/seu/codigo_de_embedding.py]` para entender como os vetores são criados.
*   **Interação com `pgvector`:** O arquivo/módulo `[caminho/absoluto/para/seu/codigo_de_db_interacao.py]` contém as funções para inserir e consultar vetores no PostgreSQL.
*   **Lógica Principal da Aplicação:** O script `[caminho/absoluto/para/seu/script_principal.py]` orquestra o fluxo.
