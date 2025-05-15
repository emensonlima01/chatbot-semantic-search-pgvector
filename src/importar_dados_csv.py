import psycopg2
import decimal
import numpy as np
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import csv
from unidecode import unidecode

load_dotenv()

# --- Configuração do Modelo de Embedding ---
try:
    print("Carregando modelo de embedding...")
    embedding_model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo de embedding: {e}")
    exit()

def get_embedding(text: str) -> list[float]:
    if not text: return [0.0] * 768
    if embedding_model: return embedding_model.encode(text).tolist()
    raise ValueError("Modelo de embedding não carregado.")

# --- Configuração da Conexão com o Banco de Dados ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "empresa1")
DB_USER = os.getenv("DB_USER", "emenson")
DB_PASSWORD = os.getenv("DB_PASSWORD", "teste123")

def get_db_connection():
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        register_vector(conn)
        return conn
    except psycopg2.Error as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        exit()

def parse_validade_string(validade_str: str):
    today = date.today()
    if not validade_str or validade_str.lower() == "indeterminada": return None
    try:
        d, m, y = map(int, validade_str.split('/')); return date(y, m, d)
    except ValueError: pass
    parts = validade_str.lower().split()
    if len(parts) != 2: return None
    try:
        value = int(parts[0]); unit = parts[1]
        if "dia" in unit: return today + timedelta(days=value)
        elif "mes" in unit or "mês" in unit: return today + relativedelta(months=value)
        elif "ano" in unit: return today + relativedelta(years=value)
        else: return None
    except ValueError: return None

def remover_acentos(text: str) -> str:
    if text is None: return ""
    return unidecode(text)

def criar_tabelas_se_nao_existirem(cur, conn_obj):
    print("Verificando/Criando extensão unaccent...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
    conn_obj.commit()
    print("Verificando/Criando tabelas...")
    cur.execute("""CREATE TABLE IF NOT EXISTS secoes_catalogo (id SERIAL PRIMARY KEY, nome TEXT NOT NULL UNIQUE, embedding VECTOR(768), secao_pai_id INTEGER REFERENCES secoes_catalogo(id) DEFAULT NULL );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS marcas (id SERIAL PRIMARY KEY, nome TEXT NOT NULL UNIQUE);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS itens_secao (id SERIAL PRIMARY KEY, secao_id INTEGER REFERENCES secoes_catalogo(id), nome TEXT NOT NULL, descricao TEXT, preco NUMERIC(10, 2), validade DATE, marca_id INTEGER REFERENCES marcas(id), embedding VECTOR(768) );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS palavras_chave_intencao (id SERIAL PRIMARY KEY, codigo_intencao TEXT NOT NULL, tipo_palavra_chave TEXT NOT NULL, valor_palavra_chave TEXT NOT NULL, prioridade INTEGER DEFAULT 0, ativo BOOLEAN NOT NULL DEFAULT TRUE, descricao TEXT );""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_palavras_chave_intencao_codigo_tipo ON palavras_chave_intencao (codigo_intencao, tipo_palavra_chave, prioridade DESC);""")
    print("Tabelas verificadas/criadas.")

def limpar_dados_existentes(cur):
    print("Limpando dados existentes...")
    cur.execute("DELETE FROM itens_secao;")
    cur.execute("DELETE FROM palavras_chave_intencao;") # Limpa também as palavras-chave
    cur.execute("DELETE FROM secoes_catalogo WHERE secao_pai_id IS NOT NULL;")
    cur.execute("DELETE FROM secoes_catalogo WHERE secao_pai_id IS NULL;")
    cur.execute("DELETE FROM marcas;")
    print("Dados limpos.")

def popular_palavras_chave_intencao_hardcoded(cur): # Nova função
    """Popula a tabela palavras_chave_intencao com dados 'hardcoded'."""
    print("\nPopulando tabela palavras_chave_intencao (dados internos)...")
    # cur.execute("DELETE FROM palavras_chave_intencao;") # Já limpo em limpar_dados_existentes

    keywords_data = [
        ("CATALOGO_GERAL", "contem", "cardápio", 0, None), ("CATALOGO_GERAL", "contem", "menu", 0, None),
        ("CATALOGO_GERAL", "contem", "catálogo", 0, None), ("CATALOGO_GERAL", "contem", "catalogo", 0, None),
        ("CATALOGO_GERAL", "contem", "lista de opções", 0, None), ("CATALOGO_GERAL", "contem", "seções", 0, None),
        ("CATALOGO_GERAL", "contem", "categorias", 0, None), ("CATALOGO_GERAL", "contem", "secoes", 0, None),
        ("CATALOGO_GERAL", "contem", "opcoes", 0, None),
        ("LISTAR_ITENS_TIPO_MARCA", "prefixo", "me diga todos os", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "prefixo", "quais são os", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "prefixo", "liste os", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "prefixo", "mostre os", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "prefixo", "todos os", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "prefixo", "quero ver os", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "prefixo", "quero os", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "separador", " da marca ", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "separador", " do marca ", 10, None),
        ("LISTAR_ITENS_TIPO_MARCA", "separador", " da ", 5, None),
        ("LISTAR_ITENS_TIPO_MARCA", "separador", " do ", 5, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "o que tem na seção", 10, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "oque tem na seção", 10, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "o que tem na", 9, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "oque tem na", 9, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "o que tem em", 8, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "oque tem em", 8, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "itens da", 5, None), ("LISTAR_ITENS_CATEGORIA", "prefixo", "itens de", 5, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "produtos da", 4, None), ("LISTAR_ITENS_CATEGORIA", "prefixo", "produtos de", 4, None),
        ("LISTAR_ITENS_CATEGORIA", "prefixo", "mostre", 0, None), ("LISTAR_ITENS_CATEGORIA", "prefixo", "liste", 0, None),
        ("LISTAR_ITENS_MARCA", "prefixo", "o que tem da marca", 10, None),
        ("LISTAR_ITENS_MARCA", "prefixo", "o que tem da", 9, None), ("LISTAR_ITENS_MARCA", "prefixo", "da marca", 6, None),
        ("LISTAR_ITENS_MARCA", "prefixo", "da", 5, None),
        ("LISTAR_ITENS_TIPO_GENERICO", "prefixo", "o que tem de", 10, None),
        ("LISTAR_ITENS_TIPO_GENERICO", "prefixo", "tem de", 9, None), ("LISTAR_ITENS_TIPO_GENERICO", "prefixo", "tem", 5, None),
        ("LISTAR_MARCAS_POR_TIPO", "prefixo", "marcas de", 10, None), ("LISTAR_MARCAS_POR_TIPO", "prefixo", "marca de", 9, None),
        ("LISTAR_MARCAS_POR_TIPO", "prefixo", "fabricantes de", 8, None), ("LISTAR_MARCAS_POR_TIPO", "prefixo", "fabricante de", 7, None)
    ]
    for codigo_intencao, tipo_kw, valor_kw, prioridade_kw, desc_kw in keywords_data:
        cur.execute(
            "INSERT INTO palavras_chave_intencao (codigo_intencao, tipo_palavra_chave, valor_palavra_chave, prioridade, descricao, ativo) VALUES (%s, %s, %s, %s, %s, TRUE);",
            (codigo_intencao, tipo_kw, valor_kw, prioridade_kw, desc_kw)
        )
    print(f"{len(keywords_data)} palavras-chave internas inseridas em palavras_chave_intencao.")

def get_or_create_secao_id(cur, nome_secao, nome_secao_pai=None, cache_secoes=None):
    """Obtém ou cria uma seção/subseção, retornando seu ID."""
    if not nome_secao: return None
    if cache_secoes is None: cache_secoes = {} # nome_normalizado -> id

    nome_secao_norm = remover_acentos(nome_secao.lower())
    if nome_secao_norm in cache_secoes:
        return cache_secoes[nome_secao_norm]

    secao_pai_id_val = None
    if nome_secao_pai:
        nome_secao_pai_norm = remover_acentos(nome_secao_pai.lower())
        if nome_secao_pai_norm in cache_secoes:
            secao_pai_id_val = cache_secoes[nome_secao_pai_norm]
        else: # Tenta buscar no banco se o pai não está no cache (deveria ter sido processado antes se é pai)
            cur.execute("SELECT id FROM secoes_catalogo WHERE lower(unaccent(nome)) = %s", (nome_secao_pai_norm,))
            res_pai = cur.fetchone()
            if res_pai: secao_pai_id_val = res_pai[0]
            else: # Cria o pai se ele realmente não existe
                print(f"  AVISO: Seção pai '{nome_secao_pai}' não encontrada, criando como principal.")
                try:
                    embedding_pai = get_embedding(nome_secao_pai)
                    cur.execute("INSERT INTO secoes_catalogo (nome, embedding, secao_pai_id) VALUES (%s, %s, NULL) ON CONFLICT (nome) DO UPDATE SET embedding = EXCLUDED.embedding RETURNING id;", (nome_secao_pai, np.array(embedding_pai)))
                    secao_pai_id_val = cur.fetchone()[0]
                    cache_secoes[nome_secao_pai_norm] = secao_pai_id_val
                    print(f"    Seção pai '{nome_secao_pai}' criada com ID: {secao_pai_id_val}")
                except psycopg2.Error as e_pai:
                    print(f"    ERRO CRÍTICO ao criar seção pai '{nome_secao_pai}': {e_pai}")
                    return None # Não pode prosseguir sem o pai

    try:
        embedding_secao = get_embedding(nome_secao)
        cur.execute(
            "INSERT INTO secoes_catalogo (nome, embedding, secao_pai_id) VALUES (%s, %s, %s) ON CONFLICT (nome) DO UPDATE SET embedding = EXCLUDED.embedding, secao_pai_id = COALESCE(EXCLUDED.secao_pai_id, secoes_catalogo.secao_pai_id) RETURNING id;",
            (nome_secao, np.array(embedding_secao), secao_pai_id_val)
        )
        secao_id = cur.fetchone()[0]
        cache_secoes[nome_secao_norm] = secao_id
        # print(f"    Seção '{nome_secao}' (Pai ID: {secao_pai_id_val}) ID: {secao_id} inserida/atualizada.")
        return secao_id
    except psycopg2.Error as e:
        print(f"    Erro ao inserir/obter seção '{nome_secao}': {e}")
        return None

def get_or_create_marca_id(cur, nome_marca, cache_marcas=None):
    """Obtém ou cria uma marca, retornando seu ID."""
    if not nome_marca: return None
    if cache_marcas is None: cache_marcas = {} # nome_normalizado -> id

    nome_marca_norm = remover_acentos(nome_marca.lower())
    if nome_marca_norm in cache_marcas:
        return cache_marcas[nome_marca_norm]

    try:
        cur.execute("INSERT INTO marcas (nome) VALUES (%s) ON CONFLICT (nome) DO NOTHING RETURNING id;", (nome_marca,))
        res = cur.fetchone()
        if res:
            marca_id = res[0]
        else: # Marca já existia
            cur.execute("SELECT id FROM marcas WHERE nome = %s;", (nome_marca,))
            marca_id = cur.fetchone()[0]
        cache_marcas[nome_marca_norm] = marca_id
        return marca_id
    except psycopg2.Error as e:
        print(f"    Erro ao inserir/obter marca '{nome_marca}': {e}")
        return None

def importar_catalogo_completo_csv(cur, conn_obj, caminho_csv='catalogo_completo.csv'):
    print(f"\nImportando catálogo completo de {caminho_csv}...")
    itens_importados_count = 0
    secoes_criadas_cache = {} # nome_normalizado -> id
    marcas_criadas_cache = {} # nome_normalizado -> id

    try:
        with open(caminho_csv, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader, 1):
                nome_item = row.get('nome_item', '').strip()
                nome_secao_item_csv = row.get('nome_secao_item', '').strip()
                nome_secao_pai_item_csv = row.get('nome_secao_pai_item', '').strip() # Pode ser vazio
                nome_marca_item_csv = row.get('nome_marca_item', '').strip() # Pode ser vazio

                desc_base = row.get('descricao_base_item', '').strip()
                outros_detalhes = row.get('outros_detalhes_item', '').strip()
                preco_str = row.get('preco_item', '').strip()
                validade_str = row.get('validade_str_item', '').strip()

                if not nome_item or not nome_secao_item_csv:
                    print(f"  Linha {i+1}: Nome do item ou nome da seção do item faltando. Pulando.")
                    continue

                # Garantir que a seção pai (se existir) seja processada primeiro
                secao_pai_id_para_item = None
                if nome_secao_pai_item_csv:
                    secao_pai_id_para_item = get_or_create_secao_id(cur, nome_secao_pai_item_csv, None, secoes_criadas_cache)
                    if not secao_pai_id_para_item:
                        print(f"  Linha {i+1}: Falha ao processar seção pai '{nome_secao_pai_item_csv}' para item '{nome_item}'. Pulando item.")
                        conn_obj.rollback(); conn_obj.commit() # Tenta continuar com o próximo item
                        continue

                # Processar a seção do item (pode ser principal ou subseção)
                secao_id_para_item = get_or_create_secao_id(cur, nome_secao_item_csv, nome_secao_pai_item_csv if nome_secao_pai_item_csv else None, secoes_criadas_cache)
                if not secao_id_para_item:
                    print(f"  Linha {i+1}: Falha ao processar seção '{nome_secao_item_csv}' para item '{nome_item}'. Pulando item.")
                    conn_obj.rollback(); conn_obj.commit()
                    continue

                # Processar marca
                marca_id_para_item = None
                if nome_marca_item_csv:
                    marca_id_para_item = get_or_create_marca_id(cur, nome_marca_item_csv, marcas_criadas_cache)
                    # Não vamos pular o item se a marca falhar, apenas não associar.
                    if not marca_id_para_item: conn_obj.rollback(); conn_obj.commit()


                descricao_completa = f"{desc_base} {outros_detalhes}".strip()
                preco = decimal.Decimal(preco_str) if preco_str else None
                data_validade = parse_validade_string(validade_str)

                # Usar o nome da seção mais específica para o embedding do item
                texto_para_embedding = f"{nome_item} {descricao_completa} Marca: {nome_marca_item_csv if nome_marca_item_csv else ''} Categoria: {nome_secao_item_csv}"
                embedding_item = get_embedding(texto_para_embedding)

                try:
                    cur.execute(
                        "INSERT INTO itens_secao (secao_id, nome, descricao, preco, validade, marca_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s);",
                        (secao_id_para_item, nome_item, descricao_completa, preco, data_validade, marca_id_para_item, np.array(embedding_item))
                    )
                    itens_importados_count += 1
                except psycopg2.Error as e:
                    print(f"  Linha {i+1}: Erro ao inserir item '{nome_item}': {e}")
                    conn_obj.rollback(); conn_obj.commit()
            conn_obj.commit() # Commit final após processar todas as linhas do CSV
    except FileNotFoundError:
        print(f"  ARQUIVO NÃO ENCONTRADO: {caminho_csv}")
    except Exception as e:
        print(f"  Erro CRÍTICO ao processar {caminho_csv}: {e}")
        import traceback; traceback.print_exc()
        if conn_obj: conn_obj.rollback()
    print(f"  {itens_importados_count} itens importados de {caminho_csv}.")


def popular_banco_via_csv_unico():
    """Função principal para importar todos os dados de um CSV único e popular palavras-chave."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        criar_tabelas_se_nao_existirem(cur, conn)
        limpar_dados_existentes(cur)
        conn.commit()

        # Popula palavras-chave internas primeiro
        popular_palavras_chave_intencao_hardcoded(cur)
        conn.commit()

        # Importa o catálogo completo do CSV
        importar_catalogo_completo_csv(cur, conn, caminho_csv='catalogo_completo.csv')
        # Não precisa mais de secoes_map como argumento, pois a função de catálogo lida com isso internamente.

        print("\nImportação de todos os dados concluída com sucesso!")

    except psycopg2.Error as e:
        print(f"Erro crítico durante a importação via CSV: {e}")
        if conn: conn.rollback()
    except Exception as e:
        print(f"Um erro geral inesperado ocorreu durante a importação: {e}")
        import traceback
        traceback.print_exc()
        if conn: conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()
            print("Conexão com o banco de dados fechada.")

if __name__ == "__main__":
    popular_banco_via_csv_unico()
